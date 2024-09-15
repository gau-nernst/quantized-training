import argparse
import json

import torch
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM

from llm_pretrain import get_loss, get_tinystories
from train_utils import print_model_stats, quantize_model

# TODO: need update
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # default config is 470M
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--ffn_size", type=int, default=4096)
    parser.add_argument("--head_dim", type=int, default=64)

    parser.add_argument("--int8_mixed_precision", type=json.loads)
    parser.add_argument("--int8_quantized_training", type=json.loads)
    parser.add_argument("--quantize_lm_head", action="store_true")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = LlamaConfig(
        hidden_size=args.d_model,
        intermediate_size=args.ffn_size,
        num_hidden_layers=args.depth,
        num_attention_heads=args.d_model // args.head_dim,
        max_position_embeddings=args.seq_len,
        use_cache=False,
    )
    model = LlamaForCausalLM(config).bfloat16().cuda()
    state_dict = torch.load(args.checkpoint, map_location="cpu", mmap=True)
    model.load_state_dict(state_dict["model"])

    quantize_model(model.model, args.int8_mixed_precision, args.int8_quantized_training)
    if args.quantize_lm_head:
        quantize_model(model.lm_head, args.int8_mixed_precision, args.int8_quantized_training)
    print_model_stats(model)

    data = get_tinystories("valid").cuda()

    total_loss = 0
    n_batches = 0
    model.eval()
    for i in tqdm(range(0, data.shape[0] - args.seq_len + 1, args.seq_len), dynamic_ncols=True):
        # fast validation loss
        # the more correct way is to calculate loss for each token with full seq_len context (rolling window)
        batch = data[i : i + args.seq_len].view(1, args.seq_len).long()

        with torch.no_grad():
            total_loss += torch.compile(get_loss)(model, batch)
        n_batches += 1

    print(f"Validation loss: {total_loss / n_batches:.4f}")
