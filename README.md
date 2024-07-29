# Quantized model training

Install PyTorch following the [official instructions](https://pytorch.org/). Recommended to use nightly version.

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch-nightly -c nvidia
```

Install other deps

```
pip install -r requirements.txt
```

## ViT fine-tuning

```
python finetune_timm.py --model timm/vit_giant_patch14_dinov2.lvd142m --n_epochs 2 --batch_size 64 --model_kwargs '{"img_size":224}' --run_name int8_model_8bit_optim --seed 2024 --model_quantize int8 --optim low_bit_optim.Adam8bit
```

## LLM fine-tuning

```
python finetune_llm.py --batch_size 4 --optim torch.optim.AdamW --ckpt_interval 10_000 --run_name qwen2_0.5b_bf16_optim8bit --n_steps 100_000 --seed 2024 --lr 1e-5 --dataset meta-math/MetaMathQA --question_key query --answer_key response --split train
```
