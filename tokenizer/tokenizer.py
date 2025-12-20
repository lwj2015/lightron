from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, model_path: str):
        # 自动加载 HF 的 tokenizer.model 或 tokenizer.json
        self.processor = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        # 确保有 pad_token，Llama 通常没有
        if self.processor.pad_token is None:
            self.processor.pad_token = self.processor.eos_token

    def encode(self, text: str, bos: bool = True, eos: bool = False):
        tokens = self.processor.encode(text, add_special_tokens=False)
        if bos:
            tokens = [self.processor.bos_token_id] + tokens
        if eos:
            tokens = tokens + [self.processor.eos_token_id]
        return tokens

    def decode(self, tokens: list) -> str:
        return self.processor.decode(tokens)

    @property
    def vocab_size(self):
        return self.processor.vocab_size
