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

4070Ti SUPER. Speedup over PyTorch BF16 matmul. See [`benchmark_mm.py`](benchmark_mm.py) (might need better configs for FP16. Use default Cutlass INT4 GEMM)

TODO: add A100 results

Row-major x Column-major (`A @ B.T`)

|                                |   1024 |   2048 |   4096 |
|:-------------------------------|-------:|-------:|-------:|
| CuBLAS INT8                    |   1.95 |   2.01 |   2.90 |
| Triton INT8                    |   2.72 |   2.87 |   3.14 |
| Cutlass INT4                   |   2.56 |   3.81 |   5.89 |
| Triton FP8                     |   1.78 |   1.65 |   1.64 |
| Triton FP16 w/ FP16 accumulate |   1.86 |   1.76 |   1.29 |

Row-major x Row-major (`A @ B`)

|                                |   1024 |   2048 |   4096 |
|:-------------------------------|-------:|-------:|-------:|
| CuBLAS INT8                    |   1.03 |   0.94 |   0.92 |
| Triton INT8                    |   1.62 |   1.98 |   2.18 |
| Triton FP8                     |   1.70 |   1.63 |   1.71 |
| Triton FP16 w/ FP16 accumulate |   1.64 |   1.77 |   1.38 |

Column-major x Row-major (`A.T @ B`)

|                                |   1024 |   2048 |   4096 |
|:-------------------------------|-------:|-------:|-------:|
| CuBLAS INT8                    |   0.87 |   0.93 |   0.88 |
| Triton INT8                    |   1.31 |   1.43 |   1.54 |
| Triton FP8                     |   1.48 |   1.61 |   1.70 |
| Triton FP16 w/ FP16 accumulate |   1.42 |   1.79 |   1.35 |

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
