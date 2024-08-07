import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import math
from datetime import datetime
from functools import partial
from pathlib import Path

import datasets
import timm
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from train_utils import get_grad_norm, get_optim_cls, print_model_stats, quantize_model


class CosineSchedule:
    def __init__(self, lr: float, total_steps: int, warmup: float = 0.05) -> None:
        self.lr = lr
        self.final_lr = 0
        self.total_steps = total_steps
        self.warmup_steps = round(total_steps * warmup)

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.lr * step / self.warmup_steps
        if step < self.total_steps:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.final_lr + 0.5 * (self.lr - self.final_lr) * (1 + math.cos(progress * math.pi))
        return self.final_lr


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_kwargs", type=json.loads, default=dict())
    parser.add_argument("--model_quantize")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--limit_steps", type=int, default=0)

    parser.add_argument("--optim", default="torch.optim.AdamW")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--optim_kwargs", type=json.loads, default=dict())
    parser.add_argument("--cosine_lr_scheduler", action="store_true")

    parser.add_argument("--project")
    parser.add_argument("--run_name", default="debug")
    parser.add_argument("--seed", type=int)
    return parser


def get_dloader(args, training: bool):
    transforms = [v2.ToImage()]

    if training:
        transforms.extend([v2.RandomResizedCrop(224), v2.RandomHorizontalFlip()])
    else:
        transforms.extend([v2.Resize(256), v2.CenterCrop(224)])

    transforms.append(v2.ToDtype(torch.float32, scale=True))
    transforms.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transforms = v2.Compose(transforms)

    ds = datasets.load_dataset("timm/resisc45", split="train" if training else "validation")
    ds = ds.select_columns(["image", "label"])
    ds.set_transform(lambda x: dict(image=transforms(x["image"]), label=x["label"]))

    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.n_workers,
        pin_memory=training,
        drop_last=training,
    )


def model_loss(model, images, labels):
    return F.cross_entropy(model(images), labels)


def model_predict(model, images):
    return model(images).argmax(1)


@torch.no_grad()
def evaluate_model(model, args):
    model.eval()
    val_dloader = get_dloader(args, False)

    all_labels = []
    all_preds = []

    for batch in tqdm(val_dloader, dynamic_ncols=True, desc=f"Evaluating"):
        all_labels.append(batch["label"].clone())

        images = batch["image"].to(dtype=torch.bfloat16, device="cuda")
        all_preds.append(torch.compile(model_predict)(model, images).cpu())

    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    acc = (all_labels == all_preds).float().mean()
    return acc


if __name__ == "__main__":
    args = get_parser().parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.limit_steps:
        args.n_epochs = 1

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    dloader = get_dloader(args, True)
    print(f"Train dataset: {len(dloader.dataset):,} images")

    model = timm.create_model(args.model, pretrained=True, num_classes=45, **args.model_kwargs)
    model.bfloat16().cuda()
    model.set_grad_checkpointing()
    quantize_model(model, args.quantize_model)
    print_model_stats(model)

    optim_cls = get_optim_cls(args.optim)
    optim = optim_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **args.optim_kwargs)
    lr_schedule = CosineSchedule(args.lr, len(dloader) * args.n_epochs)

    save_dir = Path("runs") / f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project=args.project, name=args.run_name, config=args, dir=save_dir)

    step = 0
    for epoch_idx in range(args.n_epochs):
        model.train()
        pbar = tqdm(dloader, dynamic_ncols=True, desc=f"Epoch {epoch_idx + 1}/{args.n_epochs}")

        for batch in pbar:
            loss_fn = torch.compile(model_loss) if args.compile else model_loss
            loss = loss_fn(model, batch["image"].cuda().bfloat16(), batch["label"].cuda())
            loss.backward()

            if args.cosine_lr_scheduler:
                lr = lr_schedule.get_lr(step)
                for param_group in optim.param_groups:
                    param_group["lr"] = lr

            if step % 10 == 0:
                log_dict = dict(
                    loss=loss.item(),
                    grad_norm=get_grad_norm(model),
                    lr=optim.param_groups[0]["lr"],
                )
                run.log(log_dict, step=step)
                pbar.set_postfix(loss=log_dict["loss"])

            optim.step()
            optim.zero_grad()
            step += 1

            if args.limit_steps > 0 and step == args.limit_steps:
                break

        if args.limit_steps == 0:
            val_acc = evaluate_model(model, args)
            print(f"Epoch {epoch_idx + 1}/{args.n_epochs}: val_acc={val_acc.item() * 100:.2f}")
            run.log(dict(val_acc=val_acc), step=step)

    max_memory = torch.cuda.max_memory_allocated()
    run.log(dict(max_memory=max_memory))
    print(f"Max memory allocated: {max_memory / 1e9:.2f} GiB")

    torch.save(model.state_dict(), save_dir / "model.pth")

    run.finish()
