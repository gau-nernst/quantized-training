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
python timm_finetune.py --model timm/vit_giant_patch14_dinov2.lvd142m --n_epochs 2 --batch_size 64 --model_kwargs '{"img_size":224}' --seed 2024 --model_quantize int8 --compile
```

**LLM fine-tuning on MetaMathQA**

```
python llm_finetune.py --model HuggingFaceTB/SmolLM-1.7B --freeze_embedding_layer --batch_size 4 --n_steps 100_000 --ckpt_interval 10_000 --seed 2024 --compile
```

**LLM pre-training on TinyStories**

```
python llm_tinystories_pretrain.py --seed 2024 --n_steps 100_000 --model_quantize int8
```

## Triton kernels

Benchmark INT8 matmul. 4070Ti SUPER. Speedup over PyTorch BF16 matmul.

Kernel       | `A @ B`  | `A @ B.T` | `A.T @ B`
-------------|----------|-----------|----------
`M = N = K = 1024`
PyTorch INT8 | 1.03     | 1.90      | 1.09
Triton INT8  | 1.62     | 2.54      | 1.18
`M = N = K = 2048`
PyTorch INT8 | 0.96     | 2.04      | 0.97
Triton INT8  | 1.84     | 2.91      | 1.01
`M = N = K = 4096`
PyTorch INT8 | 0.93     | 3.19      | 0.96
Triton INT8  | 2.02     | 3.25      | 1.12
