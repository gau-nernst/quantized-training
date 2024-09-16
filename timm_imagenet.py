import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from data import HFImageDataset
from train_utils import LRSchedule, get_grad_norm, get_optimizer, print_model_stats, quantize_model


def get_dloader(args, training: bool):
    transforms = [v2.ToImage()]

    if training:
        transforms.extend([v2.RandomResizedCrop(224), v2.RandomHorizontalFlip()])
    else:
        transforms.extend([v2.Resize(256), v2.CenterCrop(224)])

    transforms.append(v2.ToDtype(torch.float32, scale=True))
    transforms.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transforms = v2.Compose(transforms)

    ds = HFImageDataset(
        "timm/imagenet-1k-wds",
        split="train" if training else "validation",
        eval=not training,
        transform=transforms,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.n_workers,
        pin_memory=training,
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

    for imgs, labels in tqdm(val_dloader, dynamic_ncols=True, desc=f"Evaluating"):
        all_labels.append(labels.clone())
        all_preds.append(torch.compile(model_predict)(model, imgs.bfloat16().cuda()).cpu())

    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    acc = (all_labels == all_preds).float().mean()
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="timm/vit_tiny_patch16_224")
    parser.add_argument("--model_kwargs", type=json.loads, default=dict())

    parser.add_argument("--int8_mixed_precision", type=json.loads)
    parser.add_argument("--int8_quantized_training", type=json.loads)
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_workers", type=int, default=4)

    parser.add_argument("--optim", default="torch.optim.AdamW")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--optim_kwargs", type=json.loads, default=dict())
    parser.add_argument("--clip_grad_norm", type=int)

    parser.add_argument("--val_interval", type=int, default=1000)
    parser.add_argument("--ckpt_interval", type=int, default=1000)

    parser.add_argument("--project")
    parser.add_argument("--run_name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()
    args.torch_version = torch.__version__
    if args.seed is not None:
        torch.manual_seed(args.seed)

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    model = timm.create_model(args.model, num_classes=1000, **args.model_kwargs)
    model.bfloat16().cuda()
    model.set_grad_checkpointing()
    quantize_model(model, args.int8_mixed_precision, args.int8_quantized_training)
    print_model_stats(model)

    optim = get_optimizer(args.optim, model, args.lr, args.weight_decay, **args.optim_kwargs)
    lr_schedule = LRSchedule(args.lr, args.n_steps, 0.05, 0.1)

    args.save_dir = Path("runs/timm_imagenet") / f"{args.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    args.save_dir.mkdir(parents=True, exist_ok=True)
    logger = wandb.init(project=args.project, name=args.run_name, config=args, dir="/tmp")

    dloader_iter = iter(get_dloader(args, True))

    log_interval = 10
    step = 0
    pbar = tqdm(total=args.n_steps, dynamic_ncols=True)
    time0 = time.time()
    if args.profile:
        prof = torch.profiler.profile()

    while step < args.n_steps:
        imgs, labels = next(dloader_iter)
        loss = torch.compile(model_loss)(model, imgs.bfloat16().cuda(), labels.cuda())
        loss.backward()

        lr_schedule.set_lr(step, optim)

        if args.clip_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        else:
            grad_norm = None

        if step % log_interval == 0:
            log_dict = dict(
                loss=loss.item(),
                grad_norm=get_grad_norm(model) if grad_norm is None else grad_norm,
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
            time1 = time.time()
            log_dict = dict(
                imgs_seen=args.batch_size * step,
                imgs_per_second=args.batch_size * log_interval / (time1 - time0),
                max_memory_allocated=torch.cuda.max_memory_allocated(),
                max_memory_reserved=torch.cuda.max_memory_reserved(),
            )
            logger.log(log_dict, step=step)
            time0 = time1

        if step % args.val_interval == 0:
            val_acc = evaluate_model(model, args)
            logger.log(dict(val_acc=val_acc), step=step)

        if step % args.ckpt_interval == 0:
            ckpt = dict(
                model=model.state_dict(),
                optim=optim.state_dict(),
                step=step,
            )
            torch.save(ckpt, args.save_dir / "last.pth")

    logger.finish()
    if args.profile:
        prof.stop()
        prof.export_chrome_trace("trace.json.gz")
