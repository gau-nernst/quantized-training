# Quantized model training

This repo contains some exploration works for quantized training. Inspirations:

- Q-GaLore: [[paper](https://arxiv.org/abs/2407.08296)] [[code](https://github.com/VITA-Group/Q-GaLore)]
- AQT: [[related paper](https://arxiv.org/abs/2105.03536)] [[code](https://github.com/google/aqt)]
- SwitchBack: [[paper](https://openreview.net/forum?id=sqqASmpA2R)] [[code](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/nn/triton_based_modules.py)]
- Jetfire: [[paper](https://arxiv.org/abs/2403.12422)] [[code](https://github.com/thu-ml/Jetfire-INT8Training)]

Eventually, some of these will be upstreamed to [torchao](https://github.com/pytorch/ao).

## Environment setup

```bash
# Include submodules to clone cutlass
git clone --recurse-submodules https://github.com/gau-nernst/quantized-training
cd quantized-training

uv venv --seed --python=3.10
source .venv/bin/activate

# Install PyTorch from https://pytorch.org/. Recommend to use nightly version.
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install other deps. Might not be updated.
uv pip install -r requirements.txt
```

## Training

**LLM pre-training**

```bash
# using HF streaming dataset, tokenize on-the-fly
python llm_pretrain.py --train_ds '{"type":"hf_text","dataset":"allenai/c4","subset":"en","split":"train","tokenizer":"llama2"}' --seed 2024

# using pre-tokenized local dataset. see below. "dataset_dir" should contain .bin files
python llm_pretrain.py --train_ds '{"type":"token","dataset_dir":"tinystories_train"}' --seed 2024
```

To obtain pre-tokenized datasets, either download from [gaunernst/tokenized-datasets](https://huggingface.co/datasets/gaunernst/tokenized-datasets) or run

```bash
python tokenize_data.py --dataset tinystories --split train
```

**LLM fine-tuning on MetaMathQA**

TODO: update command

```bash
python llm_finetune.py --model HuggingFaceTB/SmolLM-1.7B --freeze_embedding_layer --batch_size 4 --n_steps 100_000 --ckpt_interval 10_000 --seed 2024 --compile
```

**ViT supervised training**

TODO ImageNet

**ViT fine-tuning on RESISC45**

TODO: update command

```bash
python timm_finetune.py --model timm/vit_giant_patch14_dinov2.lvd142m --n_epochs 2 --batch_size 64 --model_kwargs '{"img_size":224}' --seed 2024 --compile
```

## Speed benchmarks

### Matmul

RTX 5090 TFLOPS @ 400W. See [`benchmark_mm.py`](benchmark_mm.py) (might need better configs for FP16. Use default Cutlass INT4 GEMM)
- `torch==2.7.0+cu128`
- `triton==3.3.1`

Row-major x Column-major (`A @ B.T`)

|                                |   1024 |   2048 |   4096 | Theoretical |
|:-------------------------------|-------:|-------:|-------:|------------:|
| PyTorch (CuBLAS) BF16          |  87.38 | 167.72 | 176.37 |       209.5 |
| Triton FP16 w/ FP16 accumulate | 149.8  | 270.6  | 234.85 |       419   |
| Triton FP8                     | 116.51 | 188.51 | 208.41 |       419   |
| PyTorch (CuBLAS) INT8          | 210.37 | 466.03 | 479.3  |       838   |
| Triton INT8                    | 173.63 | 466.03 | 489.68 |       838   |
| Cutlass INT4                   |  17.77 |  72.42 |  74.1  |         0   |
| Inductor (Triton) scaled FP8   |  95.33 | 181.81 | 215.87 |       419   |
| Triton scaled FP8              | 116.51 | 186.41 | 207.24 |       419   |
| Triton tile-scaled FP8         |  69.91 | 158.28 | 189.57 |       419   |
| Inductor (Triton) scaled INT8  | 149.8  | 381.3  | 512.28 |       838   |
| Triton scaled INT8             | 174.76 | 493.45 | 480.56 |       838   |
| Triton tile-scaled INT8        | 149.8  | 399.46 | 399.42 |       838   |
| Cutlass scaled INT4            |  18.08 |  74.24 |  75.23 |         0   |

Row-major x Row-major (`A @ B`)

|                                |   1024 |   2048 |   4096 | Theoretical |
|:-------------------------------|-------:|-------:|-------:|------------:|
| PyTorch (CuBLAS) BF16          |  87.38 | 167.77 | 177.54 |       209.5 |
| Triton FP16 w/ FP16 accumulate | 149.8  | 270.74 | 241.36 |       419   |
| Triton FP8                     | 116.51 | 171.2  | 196.3  |       419   |
| PyTorch (CuBLAS) INT8          |  61.74 | 167.77 | 185.9  |       838   |
| Triton INT8                    | 152.52 | 363.98 | 360.8  |       838   |
| Triton scaled FP8              | 115.9  | 167.77 | 193.4  |       419   |
| Triton tile-scaled FP8         |  66.05 | 149.8  | 177.54 |       419   |
| Inductor (Triton) scaled INT8  | 131.07 | 335.54 | 413.81 |       838   |
| Triton scaled INT8             | 173.41 | 349.53 | 324.17 |       838   |
| Triton tile-scaled INT8        | 116.51 | 271.97 | 299.59 |       838   |

Column-major x Row-major (`A.T @ B`)

|                                |   1024 |   2048 |   4096 | Theoretical |
|:-------------------------------|-------:|-------:|-------:|------------:|
| PyTorch (CuBLAS) BF16          |  87.38 | 167.77 | 176.83 |       209.5 |
| Triton FP16 w/ FP16 accumulate | 149.8  | 278.17 | 244.37 |       419   |
| Triton FP8                     | 116.51 | 164.43 | 184.94 |       419   |
| PyTorch (CuBLAS) INT8          |  69.91 | 209.72 | 219.67 |       838   |
| Triton INT8                    | 147.17 | 364.72 | 362.25 |       838   |
| Triton scaled FP8              | 116.51 | 161.71 | 184.9  |       419   |
| Triton tile-scaled FP8         |  58.25 | 127.34 | 154.33 |       419   |
| Inductor (Triton) scaled INT8  | 118.15 | 226.72 | 289.1  |       838   |
| Triton scaled INT8             | 149.8  | 380.49 | 370.66 |       838   |
| Triton tile-scaled INT8        |  95.33 | 233.02 | 257.12 |       838   |

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
