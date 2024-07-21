import argparse

import torch
from lm_eval.evaluator import evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--checkpoint")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=2048)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        max_position_embeddings=args.max_seq_len,
    )

    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict)

    if args.compile:
        model.compile(mode="max-autotune", fullgraph=True)

    result = evaluate(
        HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size, max_length=args.max_seq_len),
        get_task_dict(args.tasks),
    )
    print(result)
