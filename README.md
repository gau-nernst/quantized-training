# Quantized model training

This repo contains some exploration works for quantized training. Inspirations:

- Q-GaLore: [[paper](https://arxiv.org/abs/2407.08296)] [[code](https://github.com/VITA-Group/Q-GaLore)]
- AQT: [[related paper](https://arxiv.org/abs/2105.03536)] [[code](https://github.com/google/aqt)]
- SwitchBack: [[paper](https://openreview.net/forum?id=sqqASmpA2R)] [[code](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/nn/triton_based_modules.py)]
- Jetfire: [[paper](https://arxiv.org/abs/2403.12422)] [[code](https://github.com/thu-ml/Jetfire-INT8Training)]

Eventually, some of these will be upstreamed to [torchao](https://github.com/pytorch/ao).

## Environment setup

Install PyTorch following the [official instructions](https://pytorch.org/). Recommended to use nightly version.

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch-nightly -c nvidia
```

Install other deps

```
pip install -r requirements.txt
```

## Training

**ViT fine-tuning on RESISC45**

```
python timm_finetune.py --model timm/vit_giant_patch14_dinov2.lvd142m --n_epochs 2 --batch_size 64 --model_kwargs '{"img_size":224}' --seed 2024 --compile
```

**LLM fine-tuning on MetaMathQA**

```
python llm_finetune.py --model HuggingFaceTB/SmolLM-1.7B --freeze_embedding_layer --batch_size 4 --n_steps 100_000 --ckpt_interval 10_000 --seed 2024 --compile
```

**LLM pre-training**

Prepare data: either download from [gaunernst/tokenized-datasets](https://huggingface.co/datasets/gaunernst/tokenized-datasets) or run

```
python tokenize_data.py --dataset tinystories --split train
```

Then you can run (`--dataset_dir` should contain `.bin` files)

```
python llm_pretrain.py --dataset_dir tinystories_train --seed 2024
```

## Speed benchmarks

### INT8 matmul

4070Ti SUPER. Speedup over PyTorch BF16 matmul.

Kernel       | `A @ B`  | `A @ B.T` | `A.T @ B`
-------------|----------|-----------|----------
`M = N = K = 1024`
PyTorch INT8 | 1.03     | 1.82      | 1.11
Triton INT8  | 1.63     | 2.50      | 1.56
`M = N = K = 2048`
PyTorch INT8 | 0.99     | 1.96      | 0.93
Triton INT8  | 2.11     | 2.87      | 1.44
`M = N = K = 4096`
PyTorch INT8 | 0.91     | 3.16      | 0.92
Triton INT8  | 2.21     | 3.23      | 1.53

### INT8 mixed precision training

4070Ti SUPER. Llama2-1B, bs=16, seq_len=2048. INT8 means dynamically perform row-wise quantization + scaled INT8 matmul. Exclude LM head.

Forward | Backward grad input | Backward grad weight | Stochastic rounding | tok/s  | Speedup
--------|---------------------|----------------------|---------------------|--------|--------
BF16    | BF16                | BF16                 | -                   |  9,223 | 1.00
INT8    | BF16                | BF16                 | ❌                  | 11,751 | 1.27
INT8    | BF16                | BF16                 | ✅                  | 10,944 | 1.19
INT8    | INT8                | BF16                 | ❌                  | 13,678 | 1.48
INT8    | INT8                | BF16                 | ✅                  | 12,028 | 1.30
INT8    | INT8                | INT8                 | ❌                  | 15,517 | 1.68
INT8    | INT8                | INT8                 | ✅                  | OOM

When stochastic rounding is used and backward is applied INT8 matmul, there is a significant increase in memory. To be investigated.
