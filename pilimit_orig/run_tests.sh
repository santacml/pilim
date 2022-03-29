CUDA_VISIBLE_DEVICES=0 python -m tests.test_maml&
CUDA_VISIBLE_DEVICES=1 python -m tests.test_cifar&
CUDA_VISIBLE_DEVICES=2 python -m tests.test_pimlp&