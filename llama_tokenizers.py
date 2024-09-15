import sentencepiece as spm
import tiktoken
from huggingface_hub import hf_hub_download
from tiktoken.load import load_tiktoken_bpe


def get_tokenizer(name: str):
    return dict(
        llama2=Llama2Tokenizer,
        llama3=Llama3Tokenizer,
    )[name]()


class Llama2Tokenizer:
    bos_id = 1
    eos_id = 2
    pad_id = 0

    def __init__(self):
        tokenizer_path = hf_hub_download("meta-llama/Llama-2-7b-chat-hf", "tokenizer.model")
        self.tokenizer = spm.SentencePieceProcessor(tokenizer_path)

    def __call__(self, text: str, add_bos: bool = False, add_eos: bool = False):
        return self.tokenizer.Encode(text, add_bos=add_bos, add_eos=add_eos)

    def decode(self, tokens: list[int]):
        return self.tokenizer.Decode(tokens)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size()


# https://github.com/pytorch/torchtune/blob/main/torchtune/models/llama3/_tokenizer.py
class Llama3Tokenizer:
    bos_id = 128_000
    eos_id = 128_001
    pad_id = 128_004

    def __init__(self):
        tokenizer_path = hf_hub_download("meta-llama/Meta-Llama-3.1-8B-Instruct", "original/tokenizer.model")
        pat_str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        self.tokenizer = tiktoken.Encoding(
            "llama3",
            pat_str=pat_str,
            mergeable_ranks=load_tiktoken_bpe(tokenizer_path),
            # we need to define this to decode these tokens
            special_tokens={
                "<|begin_of_text|>": 128000,
                "<|end_of_text|>": 128001,
                "<|finetune_right_pad_id|>": 128004,
            },
        )

    def __call__(self, text: str, add_bos: bool = False, add_eos: bool = False):
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        tokens.extend(self.tokenizer.encode(text, disallowed_special=()))
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens: list[int]):
        return self.tokenizer.decode(tokens)

    @property
    def vocab_size(self):
        return self.tokenizer.max_token_value + 1
