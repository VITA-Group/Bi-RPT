# Cross-Quality Few-Shot Transfer for Alloy Yield Strength Prediction: A New Materials Science Benchmark and A Sparsity-Oriented Optimization Framework

## Environment

```bash
conda env create -f environment.yml
conda activate birpt
```
## Datasets

The dataset can be found in [this link](https://drive.google.com/drive/folders/1ZBGWHfbt0NRkGd_gzxAGaP3xj-chnQ4m?usp=sharing). It can be also found in [this link](https://library.ucsd.edu/dc/object/bb8474996f).

## Experiments

### Proof-of-concept Experiments

Experiments on CUB:

```bash
cd code/proof-of-concept-experiments
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python -u cub_finetune_gradient_unroll.py --imagenet_train_data <path-to-imagenet-train-data> --imagenet_val_data <path-to-imagenet-val-data> --data data/ --rate 0.2 --save_dir cub_unroll_lr_3.5 --epoch 95 --worker 16 --dist-url tcp://127.0.0.1:37703 --lamb 1e-4 --reg-lr 3.5 --imagenet-pretrained --lower_steps 1 --seed 1 --sign-lr 1e-4
```

Experiments on CUB (10-shots):

```bash
cd code/proof-of-concept-experiments
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python -u cub_finetune_gradient_unroll.py --imagenet_train_data <path-to-imagenet-train-data> --imagenet_val_data <path-to-imagenet-val-data> --data data/ --rate 0.2 --save_dir cub_unroll_lr_3.5 --epoch 95 --worker 16 --dist-url tcp://127.0.0.1:37703 --lamb 1e-4 --reg-lr 3.5 --imagenet-pretrained --lower_steps 1 --seed 1 --sign-lr 1e-4 --ten-shot
```
