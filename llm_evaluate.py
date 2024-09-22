import argparse
import json

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from train_utils import quantize_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="mini_llamas/Llama-2-470m")
    parser.add_argument("--checkpoint")

    parser.add_argument("--quantize")
    parser.add_argument("--quantize_kwargs", type=json.loads, default=dict())
    parser.add_argument("--quantize_lm_head", action="store_true")

    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    if args.checkpoint is None:
        # load pre-trained weights
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            max_position_embeddings=args.seq_len,
            use_cache=False,
            torch_dtype=torch.bfloat16,
        )
        model.cuda()

        quantize_model(model.model, args.quantize, **args.quantize_kwargs)
        if args.quantize_lm_head:
            quantize_model(model.lm_head, args.quantize, **args.quantize_kwargs)

    else:
        # don't load pre-trained weights
        config = AutoConfig.from_pretrained(
            args.model_id,
            max_position_embeddings=args.seq_len,
            use_cache=False,
            torch_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_config(config).cuda()

        quantize_model(model.model, args.quantize, **args.quantize_kwargs)
        if args.quantize_lm_head:
            quantize_model(model.lm_head, args.quantize, **args.quantize_kwargs)

        # load weights from checkpoint, after quantization, since BitNet requires model modification
        state_dict = torch.load(args.checkpoint, map_location="cpu", mmap=True)
        model.load_state_dict(state_dict["model"])

    result = lm_eval.simple_evaluate(
        model=HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size, max_length=args.seq_len),
        tasks=args.tasks,
        limit=10 if args.debug else None,
    )
    print(result["results"])
