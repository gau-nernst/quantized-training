import argparse

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from subclass import quantize_linear_weight_int4, quantize_linear_weight_int8

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--model_quantize")
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

    if args.model_quantize == "int8":
        quantize_linear_weight_int8(model.get_decoder())
    elif args.model_quantize == "int4":
        quantize_linear_weight_int4(model.get_decoder())
    elif args.model_quantize is not None:
        raise ValueError(f"Unsupported {args.model_quantize=}")

    model.to("cuda")

    result = lm_eval.simple_evaluate(
        model=HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size, max_length=args.max_seq_len),
        tasks=args.tasks,
        limit=10 if args.debug else None,
    )
    print(result["results"])
