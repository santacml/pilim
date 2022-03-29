for each random combination of the following:
step_size: 0.5 * 2**np.arange(-2, 2, 0.5)
meta_lr: 16 * 2**np.arange(-2, 2, 0.5)
grad_clip: 0.1 * 2**np.arange(-2, 2, 0.5)
bias_alpha: 4 * 2**np.arange(-2, 2, 0.5)
first_layer_alpha: 2 * 2**np.arange(-2, 2, 0.5)
first_layer_lr_mult: 0.2 * 2**np.arange(-2, 2, 0.5)

do a sweep of the entire grid of
r in 25 * 2**np.arange(0, 4)

python -m meta.train dataset --dataset omniglot --num-ways 5 --num-shots 1 --use-cuda --step-size $step_size --batch-size 8 --num-workers 2 --num-epochs 50 --output-folder results --meta-lr $meta_lr  --grad-clip $grad_clip --meta-momentum 0 --num-shots-test 1 --normalize None --hidden-size -1 --bias-alpha $bias_alpha --infnet_r $r --first-layer-alpha $first_layer_alpha --depth 2 --verbose --first-layer-lr-mult $first_layer_lr_mult --dtype float16 --num-batches 1000 --num-test-batches 500 --adapt-readout-only --Gproj-inner --Gproj-outer --last-layer-lr-mult 0 --scheduler cosine --readout-zero-init
