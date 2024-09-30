# torchrun --standalone --nproc_per_node=2 llm_pretrain.py

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch import Tensor, nn
from torch.distributed._composable.fsdp import fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from data import get_dataset
from hellaswag import evaluate_hellaswag
from train_utils import LRSchedule, get_grad_norm, get_optimizer, print_model_stats, quantize_model


def get_loss(model: LlamaForCausalLM, tokens: Tensor, labels: Tensor):
    # last_hidden_state = model.model(tokens)[0]
    # last_hidden_state = last_hidden_state.view(-1, last_hidden_state.shape[-1])
    # logits = model.lm_head(last_hidden_state).float()
    # return F.cross_entropy(logits, labels.view(-1))
    logits = model(tokens).logits.float()
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))


# maintain FP32 weight. cast weight to BF16 to simulate inference/FSDP mixed-precision training
# instead of casting output to BF16.
class RMSNormFp32(nn.RMSNorm):
    def forward(self, x: Tensor):
        assert self.weight.dtype is torch.float32
        return F.rms_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.eps)

    @staticmethod
    def convert_llama_rmsnorm(module: nn.Module):
        if isinstance(module, LlamaRMSNorm):
            m = RMSNormFp32(module.weight.shape, module.variance_epsilon, device=module.weight.device)
            m.weight.data.copy_(module.weight)
            return m

        for name, child in module.named_children():
            setattr(module, name, RMSNormFp32.convert_llama_rmsnorm(child))
        return module


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="mini_llamas/Llama-2-470m")
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--quantize")
    parser.add_argument("--quantize_kwargs", type=json.loads, default=dict())
    parser.add_argument("--quantize_lm_head", action="store_true")
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--ddp", action="store_true")

    parser.add_argument("--train_ds", type=json.loads, required=True)
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

    parser.add_argument("--hellaswag", action="store_true")
    parser.add_argument("--hellaswag_tokenizer", default="llama2")
    parser.add_argument("--hellaswag_interval", type=int, default=1000)

    parser.add_argument("--ckpt_interval", type=int, default=1000)
    parser.add_argument("--project")
    parser.add_argument("--run_name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    is_dist = "RANK" in os.environ
    is_ddp = is_dist and args.ddp
    is_fsdp = is_dist and not args.ddp
    is_master = rank == 0
    world_size = 1
    if is_dist:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)

        if is_master:
            print(f"Using distributed training with {world_size=}")

    assert args.batch_size % (args.gradient_accumulation * world_size) == 0
    if args.seed is not None:
        torch.manual_seed(args.seed + rank)
    if args.profile:
        args.n_steps = 5
    args.torch_version = torch.__version__

    kwargs = dict(
        pretrained_model_name_or_path=args.model_id,
        max_position_embeddings=args.seq_len,
        use_cache=False,
    )
    if args.pretrained:
        model = LlamaForCausalLM.from_pretrained(**kwargs)
    else:
        model = LlamaForCausalLM(LlamaConfig.from_pretrained(**kwargs))
    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    # keep RoPE cache in FP32
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding, LlamaRMSNorm)):
            m.bfloat16()
    assert model.model.rotary_emb.inv_freq.dtype is torch.float32
    model.cuda()

    quantize_model(model.model, args.quantize, **args.quantize_kwargs)
    if args.quantize_lm_head:
        quantize_model(model.lm_head, args.quantize, **args.quantize_kwargs)

    # do this after quantization because BitNet will add more RMSNorm layers
    # this will force RMSNorm weights to be in FP32, since it might be underflow
    # during weight update if it is BF16.
    RMSNormFp32.convert_llama_rmsnorm(model)

    if is_ddp:
        # not compatible with activation checkpointing https://github.com/pytorch/pytorch/issues/104674
        # gradients all-reduce won't overlap with backward. however, the speedup thanks to full
        # compiled graph outweighs comm overlap.
        if args.activation_checkpointing:
            torch._dynamo.config.optimize_ddp = False
        model = DDP(model)

    elif is_fsdp:
        # TODO: reduce in FP32?
        # need to fully_shard RMSNorm separately since it's FP32
        # TODO: add optimization from awgu https://github.com/awgu/torchtrain/commits/fsdp_rmsnorm/
        for module in model.modules():
            if isinstance(module, RMSNormFp32):
                fully_shard(module)
        for layer in model.model.layers:
            fully_shard(layer)
            layer.compile()  # FSDP is more performant when compiling this way
        fully_shard(model)

    if is_master:
        print_model_stats(model)

    optim = get_optimizer(args.optim, model, args.lr, args.weight_decay, **args.optim_kwargs)
    if args.lr_schedule_kwargs is not None:
        lr_schedule = LRSchedule(args.lr, args.n_steps, **args.lr_schedule_kwargs)
    else:
        lr_schedule = None

    ds = get_dataset(seq_len=args.seq_len, eval=False, **args.train_ds)
    bsize = args.batch_size // (args.gradient_accumulation * world_size)
    dloader = iter(DataLoader(ds, batch_size=bsize, num_workers=args.n_workers, pin_memory=True))

    args.save_dir = Path("runs/llm_pretrain") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    if is_master:
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
    pbar = tqdm(total=args.n_steps, dynamic_ncols=True, disable=is_master == False)
    model.train()
    loss_fn = get_loss if is_fsdp else torch.compile(get_loss)
    time0 = time.time()
    if args.profile and is_master:
        torch._inductor.config.triton.unique_kernel_names = True
        prof = torch.profiler.profile()

    while step < args.n_steps:
        # TODO: disable gradient all-reduce for non-last micro-steps
        for _ in range(args.gradient_accumulation):
            tokens, labels = next(dloader)
            loss = loss_fn(model, tokens.cuda(), labels.cuda())
            loss.backward()

        if lr_schedule is not None:
            lr_schedule.set_lr(step, optim)

        if step % log_interval == 0:
            if is_dist:
                dist.all_reduce(loss, dist.ReduceOp.AVG)
            log_dict = dict(
                loss=loss.item(),
                grad_norm=get_grad_norm(model),
                lr=optim.param_groups[0]["lr"],
            )
            if is_master:
                logger.log(log_dict, step=step)
                pbar.set_postfix(loss=log_dict["loss"])

        optim.step()
        optim.zero_grad()

        step += 1
        pbar.update()
        if args.profile and step == 1 and is_master:
            prof.start()

        if step % log_interval == 0 and is_master:
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
            if is_fsdp:  # FSDP saves on all ranks
                torch.save(ckpt, args.save_dir / f"last_{rank}.pth")
            elif is_master:  # single-device or DDP - only rank 0
                torch.save(ckpt, args.save_dir / "last.pth")

        if args.hellaswag and step % args.hellaswag_interval == 0:
            if is_master:
                acc = evaluate_hellaswag(model, args.hellaswag_tokenizer)
                logger.log(dict(hellaswag_acc=acc), step=step)

            if is_dist:
                dist.barrier()
            model.train()

    if is_master:
        logger.finish()
        if args.profile:
            prof.stop()
            prof.export_chrome_trace("trace.json.gz")

    if is_dist:
        dist.destroy_process_group()
