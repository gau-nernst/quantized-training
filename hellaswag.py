import re

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor
from tqdm import tqdm
from transformers import LlamaForCausalLM

from llama_tokenizers import get_tokenizer


# https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.4/lm_eval/tasks/hellaswag/utils.py
def preprocess(text: str):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def predict(model: LlamaForCausalLM, data: Tensor) -> Tensor:
    N, n_choices, seq_len = data.shape

    inputs = data[..., :-1].reshape(N * n_choices, seq_len - 1)
    logits = model(inputs).logits.float()  # (N * n_choices, seq_len - 1, vocab_size)

    labels = data[..., 1:]  # (N, n_choices, seq_len - 1)
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.flatten(), reduction="none")

    loss = loss.reshape_as(labels).sum(-1) / (labels != -100).sum(-1)
    preds = loss.argmin(-1)
    return preds


def evaluate_hellaswag(model: LlamaForCausalLM, tokenizer: str, split: str = "validation", pbar: bool = True) -> None:
    # using Llama2 tokenizer, max 170 toks -> always pad to 256 to avoid re-compile
    ds = load_dataset("Rowan/hellaswag", split=split)
    tokens = torch.zeros(len(ds), 4, 257, dtype=torch.int64)
    tokenizer = get_tokenizer(tokenizer)

    # TODO: cache this
    for row_idx, row in enumerate(ds):
        ctx = f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}"
        for ending_idx, ending in enumerate(row["endings"]):
            toks = tokenizer(preprocess(f"{ctx} {ending}"))
            assert len(toks) <= 256
            tokens[row_idx, ending_idx, : len(toks)] = torch.tensor(toks)
            tokens[row_idx, ending_idx, len(toks) :] = -100

    n_correct = 0
    bsize = 2
    all_labels = torch.tensor([int(x) for x in ds["label"]])
    model.eval()
    for i in tqdm(range(0, len(ds), bsize), desc=f"Evaluate hellaswag {split}", disable=not pbar):
        end_idx = min(i + bsize, len(ds))
        data = tokens[i:end_idx]
        labels = all_labels[i:end_idx]

        preds = torch.compile(predict)(model, data.cuda())
        n_correct += (preds.cpu() == labels).sum()

    return n_correct / len(ds)
