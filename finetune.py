import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
from datetime import datetime
from pathlib import Path

import safetensors
import torch
import wandb
from torch import Tensor, nn
from torchao.prototype import low_bit_optim
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def _data_iter(input_ids: Tensor, attn_mask: Tensor, batch_size: int, training: bool):
    n = input_ids.shape[0]

    if not training:
        for i in range(0, n, batch_size):
            inputs = input_ids[i : min(i + batch_size, n)]
            mask = attn_mask[i : min(i + batch_size, n)]

            labels = torch.where(mask.bool(), inputs, -100)
            yield inputs.cuda().long(), labels.cuda().long()
        raise StopIteration

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


def create_dataset(data_path: str, batch_size: int, training: bool):
    with safetensors.safe_open(data_path, framework="pt") as f:
        input_ids = f.get_tensor("input_ids")
        attn_mask = f.get_tensor("attn_mask")

    return _data_iter(input_ids, attn_mask, batch_size, training), input_ids.shape[0]


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
        AdamW8bit=low_bit_optim.AdamW8bit,
        AdamWFp8=low_bit_optim.AdamWFp8,  # requires PyTorch 2.4. embedding layer should be frozen.
    )[optimizer]

    optim_dict = {p: optim_cls([p], lr=lr, weight_decay=weight_decay, **optimizer_kwargs) for p in model.parameters()}

    def optim_hook(p):
        optim_dict[p].step()
        optim_dict[p].zero_grad()

    for p in model.parameters():
        if p.requires_grad:
            p.register_post_accumulate_grad_hook(optim_hook)

    return optim_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--max_seq_length", type=int, default=2048)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=1000)

    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--optimizer_kwargs", type=json.loads, default=dict())

    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--ckpt_interval", type=int, default=1000)
    parser.add_argument("--project")
    parser.add_argument("--run_name")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

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
    model.get_input_embeddings().requires_grad_(False)
    print(f"Vocab size: {model.vocab_size:,}")
    print(f"No. of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"No. of non-trainable params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    print(f"No. of buffers: {sum(p.numel() for p in model.buffers()):,}")

    optim_dict = setup_fuse_backward_with_optim_step(
        model, args.optimizer, args.lr, args.weight_decay, args.optimizer_kwargs
    )

    ds_path = Path("tokenized_data") / args.dataset.replace("/", "_") / args.model.replace("/", "_")
    train_ds, train_size = create_dataset(ds_path / "train.safetensors", args.batch_size, True)
    print(f"Training dataset size: {train_size:,}")
    print(f"Each epoch will takes {train_size // args.batch_size:,} iters to finish")

    Path("wandb_logs").mkdir(exist_ok=True)
    wandb.init(project=args.project, name=args.run_name, dir="wandb_logs", config=args)

    ckpt_dir = Path("checkpoints") / args.dataset.replace("/", "_") / args.model.replace("/", "_")
    ckpt_dir = ckpt_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
            loss_python = loss.item()
            pbar.set_postfix(loss=loss_python)
            wandb.log(dict(loss=loss_python), step=step)

        if args.eval_interval > 0 and step % args.eval_interval == 0:
            eval_ds, eval_size = create_dataset(ds_path / "test.safetensors", args.batch_size, False)
            model.eval()
            total_loss = 0

            # not exactly accurate
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(tqdm(eval_ds, total=eval_size, desc="Eval")):
                    loss = torch.compile(get_loss, fullgraph=True)(model, inputs, labels)
                    total_loss = (total_loss * i + loss) / (i + 1)

            print(f"Eval loss: {total_loss:.4f}")
            wandb.log(dict(eval_loss=total_loss), step=step)

            model.train()

        if args.ckpt_interval > 0 and step % args.ckpt_interval == 0:
            # TODO: checkpoint optimizer
            ckpt = dict(
                model=model.state_dict(),
                step=step,
            )
            torch.save(ckpt, ckpt_dir / "last.pth")

    wandb.finish()
