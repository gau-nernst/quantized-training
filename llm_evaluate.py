import argparse
import json

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_utils import quantize_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--int8_mixed_precision", type=json.loads)
    parser.add_argument("--int8_quantized_training", type=json.loads)

    parser.add_argument("--checkpoint")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        max_position_embeddings=args.max_seq_len,
    )

    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, assign=True)
    model.to("cuda")

    quantize_model(model.get_decoder(), args.int8_mixed_precision, args.int8_quantized_training)

    result = lm_eval.simple_evaluate(
        model=HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size, max_length=args.max_seq_len),
        tasks=args.tasks,
        limit=10 if args.debug else None,
    )
    print(result["results"])
