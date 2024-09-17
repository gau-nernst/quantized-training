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

TODO: use steam-based C4 instead, like torchtitan

```
python tokenize_data.py --dataset tinystories --split train
```

Then you can run (`--dataset_dir` should contain `.bin` files)

```
python llm_pretrain.py --dataset_dir tinystories_train --seed 2024
```

## Speed benchmarks

### Matmul

4070Ti SUPER. Speedup over PyTorch BF16 matmul. See [`benchmark_mm.py`](benchmark_mm.py) (might need better configs for FP16)

Kernel       | `A @ B`  | `A @ B.T` | `A.T @ B`
-------------|----------|-----------|----------
`M = N = K = 1024`
PyTorch INT8 | 1.03     | 1.93      | 1.02
Triton INT8  | 1.70     | 2.60      | 1.56
Cutlass INT4 | -        | 2.50      | -
Triton FP8   | 1.70     | 2.17      | 1.44
Triton FP16* | 1.69     | 1.70      | 1.63
`M = N = K = 2048`
PyTorch INT8 | 0.99     | 1.99      | 0.98
Triton INT8  | 2.08     | 2.91      | 1.51
Cutlass INT4 | -        | 3.92      | -
Triton FP8   | 1.71     | 1.94      | 1.31
Triton FP16* | 1.87     | 1.80      | 1.86
`M = N = K = 4096`
PyTorch INT8 | 0.89     | 3.58      | 0.96
Triton INT8  | 2.17     | 3.12      | 1.52
Cutlass INT4 | -        | 5.89      | -
Triton FP8   | 1.70     | 1.99      | 1.30
Triton FP16* | 1.31     | 1.27      | 1.34

*: FP16 matmul with **FP16 accumulate** (NOT FP32 accumulate).

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
