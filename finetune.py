import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
from pathlib import Path

import safetensors
import torch
from torch import nn
from torchao.prototype import low_bit_optim
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def create_dataset(data_path: str, batch_size: int):
    with safetensors.safe_open(data_path, framework="pt") as f:
        input_ids = f.get_tensor("input_ids")
        attn_mask = f.get_tensor("attn_mask")

    n = input_ids.shape[0]
    print(
        f"{data_path} has {n:,} samples. With {batch_size=:,}, it takes"
        f" {n // batch_size:,} iterations to finish 1 epoch."
    )

    while True:
        # shuffle
        indices = torch.randperm(n)
        input_ids = input_ids[indices]
        attn_mask = attn_mask[indices]

        for i in range(0, n - batch_size + 1, batch_size):
            inputs = input_ids[i : i + batch_size]
            mask = attn_mask[i : i + batch_size]

            labels = torch.where(mask.bool(), inputs, -100)
            yield inputs.cuda().long(), labels.cuda().long()


def get_loss(model, inputs, labels):
    return model(inputs, labels=labels).loss


def setup_fuse_backward_with_optim_step(
    model: nn.Module,
    optimizer: str,
    lr: float,
    weight_decay: float,
    optimizer_kwargs: dict,
):
    optim_cls = dict(
        AdamW=torch.optim.AdamW,
        AdamW8bit=low_bit_optim.Adam8bit,
    )[optimizer]

    optim_dict = {p: optim_cls([p], lr=lr, weight_decay=weight_decay, **optimizer_kwargs) for p in model.parameters()}

    def optim_hook(p):
        optim_dict[p].step()
        optim_dict[p].zero_grad()

    for p in model.parameters():
        p.register_post_accumulate_grad_hook(optim_hook)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-1.5B")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--max_seq_length", type=int, default=2048)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=1000)

    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--optimizer_kwargs", type=json.loads, default=dict())

    args = parser.parse_args()

    # NOTE: must set max_position_embeddings to not store excessive positional encodings buffer
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        max_position_embeddings=args.max_seq_length,
        use_cache=False,
    )
    model.gradient_checkpointing_enable()
    print(f"Vocab size: {model.vocab_size:,}")
    print(f"Number of params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of buffers: {sum(p.numel() for p in model.buffers()):,}")

    setup_fuse_backward_with_optim_step(model, args.optimizer, args.lr, args.weight_decay, args.optimizer_kwargs)

    ds_path = Path("tokenized_data") / args.dataset.replace("/", "_") / args.model.replace("/", "_")
    train_ds = create_dataset(ds_path / "train.safetensors", args.batch_size)

    step = 0
    pbar = tqdm(total=args.n_steps, dynamic_ncols=True)
    model.train()

    while step < args.n_steps:
        inputs, labels = next(train_ds)
        loss = torch.compile(get_loss, fullgraph=True)(model, inputs, labels)
        loss.backward()

        step += 1
        pbar.update()

        if step % 50 == 0:
            pbar.set_postfix(loss=loss.item())
