import argparse
import json
from pathlib import Path

import safetensors
import torch
import torch.nn.functional as F
from torchao.prototype import low_bit_optim
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def create_dataset(data_path: str, batch_size: int):
    with safetensors.safe_open(data_path, framework="pt") as f:
        input_ids = f.get_tensor("input_ids")
        attn_mask = f.get_tensor("attn_mask")

    n = input_ids.shape[0]
    while True:
        # shuffle
        indices = torch.randperm(n)
        input_ids = input_ids[indices]
        attn_mask = attn_mask[indices]

        for i in range(0, n - batch_size + 1, batch_size):
            inputs = input_ids[i : i + batch_size, :1024]
            mask = attn_mask[i : i + batch_size, :1024]

            labels = F.pad(inputs[:, 1:], (0, 1))  # predict next token
            mask[:, -1] = 0  # don't calculate loss for last token
            labels = torch.where(mask.bool(), labels, -100)

            yield inputs.cuda().long(), labels.cuda().long()


def get_loss(model, inputs, labels):
    return F.cross_entropy(model(inputs).logits.flatten(0, 1), labels.view(-1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-1.5B")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--max_seq_length", type=int, default=2048)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation", type=int, default=1)

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
    )
    print(f"Vocab size: {model.vocab_size:,}")
    print(f"Number of params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of buffers: {sum(p.numel() for p in model.buffers()):,}")

    optim_cls = dict(
        AdamW=torch.optim.AdamW,
        AdamW8bit=low_bit_optim.Adam8bit,
    )[args.optimizer]
    optim = optim_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **args.optimizer_kwargs)

    ds_path = Path("tokenized_data") / args.dataset.replace("/", "_") / args.model.replace("/", "_")
    train_ds = create_dataset(ds_path / "train.safetensors", args.batch_size // args.gradient_accumulation)

    step = 0
    pbar = tqdm(total=args.n_steps)
    model.train()

    while step < args.n_steps:
        for _ in range(args.gradient_accumulation):
            inputs, labels = next(train_ds)
            loss = torch.compile(get_loss)(model, inputs, labels)
            (loss / args.gradient_accumulation).backward()

        optim.step()
        optim.zero_grad()
        step += 1
        pbar.update()

        if step % 50 == 0:
            pbar.set_postfix(loss=loss.item())
