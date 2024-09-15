import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import wandb
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM

from streaming_datasets import get_dataset
from train_utils import LRSchedule, get_grad_norm, get_optimizer, print_model_stats, quantize_model


def get_loss(model: LlamaForCausalLM, tokens: Tensor, labels: Tensor):
    logits = model(tokens).logits.flatten(0, 1)
    return torch.nn.functional.cross_entropy(logits, labels.view(-1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="mini_llamas/Llama-2-470m")

    parser.add_argument("--int8_mixed_precision", type=json.loads)
    parser.add_argument("--int8_quantized_training", type=json.loads)
    parser.add_argument("--quantize_lm_head", action="store_true")
    parser.add_argument("--activation_checkpointing", action="store_true")

    parser.add_argument("--train_ds", required=True)
    parser.add_argument("--train_ds_kwargs", type=json.loads, default=dict())
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--gradient_accumulation", type=int, default=1)

    parser.add_argument("--optim", default="torch.optim.AdamW")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--optim_kwargs", type=json.loads, default=dict())
    parser.add_argument("--lr_schedule_kwargs", type=json.loads)

    parser.add_argument("--val_ds", required=True)
    parser.add_argument("--val_ds_kwargs", type=json.loads, default=dict())
    parser.add_argument("--val_interval", type=int, default=1000)

    parser.add_argument("--ckpt_interval", type=int, default=1000)
    parser.add_argument("--project")
    parser.add_argument("--run_name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    assert args.batch_size % args.gradient_accumulation == 0
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.profile:
        args.n_steps = 5
    args.torch_version = torch.__version__

    config = LlamaConfig.from_pretrained(
        args.model_id,
        max_position_embeddings=args.seq_len,
        use_cache=False,
    )
    model = LlamaForCausalLM(config).bfloat16().cuda()
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()
    quantize_model(model.model, args.int8_mixed_precision, args.int8_quantized_training)
    if args.quantize_lm_head:
        quantize_model(model.lm_head, args.int8_mixed_precision, args.int8_quantized_training)
    print_model_stats(model)

    optim = get_optimizer(args.optim, model, args.lr, args.weight_decay, **args.optim_kwargs)
    if args.lr_schedule_kwargs is not None:
        lr_schedule = LRSchedule(args.lr, args.n_steps, **args.lr_schedule_kwargs)
    else:
        lr_schedule = None

    ds = get_dataset(
        args.train_ds,
        seq_len=args.seq_len,
        eval=False,
        **args.train_ds_kwargs,
    )
    bsize = args.batch_size // args.gradient_accumulation
    dloader = iter(DataLoader(ds, batch_size=bsize, num_workers=args.n_workers, pin_memory=True))

    args.save_dir = Path("runs/llm_pretrain") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    args.save_dir.mkdir(parents=True, exist_ok=True)
    logger = wandb.init(
        dir="/tmp",
        config=args,
        project=args.project,
        name=args.run_name,
        mode="disabled" if args.profile else None,
    )

    step = 0
    log_interval = 50
    pbar = tqdm(total=args.n_steps, dynamic_ncols=True)
    model.train()
    time0 = time.time()
    if args.profile:
        prof = torch.profiler.profile()

    while step < args.n_steps:
        for _ in range(args.gradient_accumulation):
            tokens, labels = next(dloader)
            loss = torch.compile(get_loss)(model, tokens.cuda(), labels.cuda())
            loss.backward()

        if lr_schedule is not None:
            lr_schedule.set_lr(step, optim)

        if step % log_interval == 0:
            log_dict = dict(
                loss=loss.item(),
                grad_norm=get_grad_norm(model),
                lr=optim.param_groups[0]["lr"],
            )
            logger.log(log_dict, step=step)
            pbar.set_postfix(loss=log_dict["loss"])

        optim.step()
        optim.zero_grad()

        step += 1
        pbar.update()
        if args.profile and step == 1:
            prof.start()

        if step % log_interval == 0:
            tokens_per_batch = args.batch_size * args.seq_len
            time1 = time.time()
            log_dict = dict(
                max_memory_allocated=torch.cuda.max_memory_allocated(),
                num_tokens_seen_millions=tokens_per_batch * step / 1e6,
                tokens_per_second=tokens_per_batch * log_interval / (time1 - time0),
            )
            time0 = time1
            logger.log(log_dict, step=step)

        if args.ckpt_interval > 0 and step % args.ckpt_interval == 0:
            ckpt = dict(
                model=model.state_dict(),
                optim=optim.state_dict(),
                step=step,
            )
            torch.save(ckpt, args.save_dir / "last.pth")

        if args.val_interval > 0 and step % args.val_interval == 0 and args.val_ds is not None:
            val_ds = get_dataset(
                args.val_ds,
                seq_len=args.seq_len,
                eval=True,
                **args.val_ds_kwargs,
            )
            val_dloader = DataLoader(val_ds, batch_size=bsize, num_workers=1)

            total_loss = 0
            n_batches = 0
            model.eval()
            with torch.no_grad():
                for tokens, labels in tqdm(val_dloader, desc="Evaluating", dynamic_ncols=True):
                    total_loss += torch.compile(get_loss)(model, tokens.cuda(), labels.cuda()).item()
                    n_batches += 1
            val_loss = total_loss / n_batches
            logger.log(dict(val_loss=val_loss), step=step)
            model.train()

    logger.finish()
    if args.profile:
        prof.stop()
        prof.export_chrome_trace("trace.json.gz")
