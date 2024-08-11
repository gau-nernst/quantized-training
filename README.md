# Quantized model training

This repo contains some exploration works for quantized training. Inspirations:

- Q-GaLore: [[paper](https://arxiv.org/abs/2407.08296)] [[code](https://github.com/VITA-Group/Q-GaLore)]
- AQT: [[related paper](https://arxiv.org/abs/2105.03536)] [[code](https://github.com/google/aqt)]

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

## ViT fine-tuning on RESISC45

```
python timm_finetune.py --model timm/vit_giant_patch14_dinov2.lvd142m --n_epochs 2 --batch_size 64 --model_kwargs '{"img_size":224}' --seed 2024 --optim low_bit_optim.AdamW --model_quantize int8 --compile
```

## LLM fine-tuning on MetaMathQA

```
python llm_finetune.py --model HuggingFaceTB/SmolLM-1.7B --freeze_embedding_layer --batch_size 4 --n_steps 100_000 --optim low_bit_optim.AdamW --ckpt_interval 10_000 --seed 2024 --compile
```

## LLM pre-training on TinyStories

```
python llm_pretrain.py --seed 2024 --n_steps 100_000 --model_quantize int8
```
