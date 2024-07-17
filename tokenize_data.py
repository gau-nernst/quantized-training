import argparse
from pathlib import Path

import safetensors.torch
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-1.5B")
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--train_split", default="train_sft")
    parser.add_argument("--test_split", default="test_sft")
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.model_max_length = args.max_length
    tokenizer.padding_side = "right"

    def apply_chat_template(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            add_generation_prompt=False,
            tokenize=True,
            padding=True,
            truncation=True,
            return_dict=True,
        )

    def tokenize_dataset(dataset: str, split: str, desc: str, save_path: Path):
        ds = load_dataset(dataset, split=split)
        column_names = list(ds.features)
        ds = ds.map(apply_chat_template, batched=True, remove_columns=column_names, desc=desc)

        data = dict(
            input_ids=torch.tensor(ds["input_ids"], dtype=torch.int32),
            attn_mask=torch.tensor(ds["attention_mask"], dtype=torch.uint8),
        )
        safetensors.torch.save_file(data, save_path)

    save_dir = Path("tokenized_data") / args.dataset.replace("/", "_") / args.model.replace("/", "_")
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenize_dataset(args.dataset, args.train_split, "Train set", save_dir / "train.safetensors")
    tokenize_dataset(args.dataset, args.test_split, "Test set", save_dir / "test.safetensors")
