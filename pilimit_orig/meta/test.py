import torch
torch.set_default_dtype(torch.float64)
import torch.nn.functional as F
import os
import json

from torchmeta.utils.data import BatchMetaDataLoader

import cox.store
from cox import utils

import argparse

from meta.maml.datasets import get_benchmark_by_name
from meta.maml.metalearners import ModelAgnosticMetaLearning, InfMAML

def main(args, store):
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.folder is not None:
        config['folder'] = args.folder
    if args.num_steps > 0:
        config['num_steps'] = args.num_steps
    if args.num_batches > 0:
        config['num_batches'] = args.num_batches
    if args.seed is not None:
        config['seed'] = args.seed
    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')
    print(config)
    conf = argparse.Namespace(**config)
    torch.set_default_dtype(getattr(torch, conf.dtype))

    benchmark = get_benchmark_by_name(conf.dataset,
                                      conf.folder,
                                      conf.num_ways,
                                      conf.num_shots,
                                      conf.num_shots_test,
                                      hidden_size=conf.hidden_size,
                                      normalize=conf.normalize,
                                      bias_alpha=conf.bias_alpha,
                                      last_bias_alpha=conf.last_bias_alpha,
                                      first_layer_alpha=conf.first_layer_alpha,
                                      depth=conf.depth,
                                      infnet_r=conf.infnet_r,
                                      readout_zero_init=conf.readout_zero_init,
                                      seed=conf.seed,
                                      orig_model=conf.orig_model,
                                      lin_model=conf.lin_model,
                                      sigma1=conf.sigma1,
                                      sigma2=conf.sigma2,
                                      sigmab=conf.sigmab,
                                      train_last_layer_only=conf.train_last_layer_only,
                                      gp1lp=conf.gp1lp,
                                      ntk1lp=conf.ntk1lp)
    # benchmark = get_benchmark_by_name(config['dataset'],
    #                                   config['folder'],
    #                                   config['num_ways'],
    #                                   config['num_shots'],
    #                                   config['num_shots_test'],
    #                                   hidden_size=config['hidden_size'],
    #                                   normalize=config['normalize'],
    #                                   bias_alpha=config['bias_alpha'],
    #                                   last_bias_alpha=config.get('last_bias_alpha', 0),
    #                                   depth=config.get('depth', 2),
    #                                   infnet_r=config.get('infnet_r', 100),
    #                                   seed=config['seed'],
    #                                   orig_model=config.get('orig_model', False),
    #                                   lin_model=config.get('lin_model', False),
    #                                   train_last_layer_only=config.get('train_last_layer_only', False),
    #                                   gp1lp=config.get('gp1lp', False),
    #                                   ntk1lp=config.get('ntk1lp', False))

    #with open(config['model_path'], 'rb') as f:
    with open(os.path.join(os.path.dirname(args.config), 'model.th'), 'rb') as f:
        benchmark.model.load_state_dict(torch.load(f, map_location=device))

    # benchmark.model.no_adapt_readout = config.get('no_adapt_readout', True)

    if args.test_dataset_split == 'val':
        dataset = benchmark.meta_val_dataset
    elif args.test_dataset_split == 'test':
        dataset = benchmark.meta_test_dataset
    else:
        raise ValueError()
    meta_test_dataloader = BatchMetaDataLoader(dataset,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=True)
    if config['hidden_size'] < 0 and not (config['lin_model'] and not config['orig_model']):
        readout_fixed_at_zero = conf.last_layer_lr_mult==0 and conf.last_bias_alpha==0 and conf.readout_zero_init
        if readout_fixed_at_zero:
            print('readout will be fixed at zero before adaptation')
        metalearner = InfMAML(benchmark.model,
                            first_order=conf.first_order,
                            num_adaptation_steps=conf.num_steps,
                            step_size=conf.step_size,
                            loss_function=benchmark.loss_function,
                            train_loss_function=conf.train_loss_function,
                            readout_fixed_at_zero=readout_fixed_at_zero,
                            # grad_clip=conf.grad_clip,
                            # gclip_per_param=conf.gclip_per_param,
                            device=device,
                            no_adapt_readout=conf.no_adapt_readout,
                            adapt_readout_only=conf.adapt_readout_only,
                            Gproj_inner=conf.Gproj_inner,
                            Gproj_outer=conf.Gproj_outer)
        # metalearner = InfMAML(benchmark.model,
        #                     first_order=config['first_order'],
        #                     num_adaptation_steps=config['num_steps'],
        #                     step_size=config['step_size'],
        #                     loss_function=benchmark.loss_function,
        #                     no_adapt_readout=config.get('no_adapt_readout', True),
        #                     device=device)

    else:
        metalearner = ModelAgnosticMetaLearning(
            benchmark.model,
            first_order=conf.first_order,
            num_adaptation_steps=conf.num_steps,
            step_size=conf.step_size,
            loss_function=benchmark.loss_function,
            train_loss_function=conf.train_loss_function,
            # grad_clip=conf.grad_clip,
            no_adapt_readout=conf.no_adapt_readout,
            adapt_readout_only=conf.adapt_readout_only,
            device=device,
            Gproj_inner=conf.Gproj_inner,
            Gproj_outer=conf.Gproj_outer)
        # metalearner = ModelAgnosticMetaLearning(benchmark.model,
        #                                     first_order=config['first_order'],
        #                                     num_adaptation_steps=config['num_steps'],
        #                                     step_size=config['step_size'],
        #                                     loss_function=benchmark.loss_function,
        #                                     device=device)

    results = metalearner.evaluate(meta_test_dataloader,
                                   max_batches=config['num_batches'],
                                   verbose=args.verbose,
                                   nilmax=args.nilmax,
                                   desc='Test')
    store[f'test_result_batch{args.num_batches}_step{args.num_steps}'].append_row(results)
    '''
    # Save results
    dirname = os.path.dirname(config['model_path'])
    with open(os.path.join(dirname, 'results.json'), 'w') as f:
        json.dump(results, f)
    '''

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')
    parser.add_argument('--config', type=str,
        help='Path to the configuration file returned by `train.py`.')
    parser.add_argument('--folder', type=str, default=None,
        help='Path to the folder the data is downloaded to. '
        '(default: path defined in configuration file).')
    parser.add_argument('--test_dataset_split', type=str, default='test',
        choices=('test', 'val'))

    # Optimization
    parser.add_argument('--num-steps', type=int, default=-1,
        help='Number of fast adaptation steps, ie. gradient descent updates '
        '(default: number of steps in configuration file).')
    parser.add_argument('--num-batches', type=int, default=-1,
        help='Number of batch of tasks per epoch '
        '(default: number of batches in configuration file).')
    parser.add_argument('--nilmax', action='store_true',
        help='Use NIL with 0 temperature for adaptation')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()

    output_folder = os.path.dirname(os.path.dirname(args.config))
    exp_id = args.config.split('/')[-2]
    print(output_folder, exp_id)
    store = cox.store.Store(output_folder, exp_id)
    assert 'finished' in store.keys

    if f'test_result_batch{args.num_batches}_step{args.num_steps}' not in store.keys:
        store.add_table(f'test_result_batch{args.num_batches}_step{args.num_steps}', {
            "mean_outer_loss": float,
            "median_outer_loss": float,
            "accuracies_after": float,
        })

    main(args, store)
