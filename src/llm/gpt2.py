import numpy as np
from tokenizers import Tokenizer
import pickle
import random
from src.llm.utils import (
    layer_norm,
    ffn,
    mha,
    gelu,
)
import os


class GPT2:
    def __init__(self, model_path=None, tokenizer_path=None, n_head=12):
        if model_path is None:
            model_path = os.getenv("GPT2_MODEL_PATH")
        if tokenizer_path is None:
            tokenizer_path = os.getenv("GPT2_TOKENIZER_PATH")
        self.n_head = n_head
        self.load_model(model_path)
        self.load_tokenizer(tokenizer_path)

    def load_model(self, model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            self.params = model["params"]
            self.hparams = model["hparams"]

    def load_tokenizer(self, tokenizer_path):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def transformer_block(self, x, mlp, attn, ln_1, ln_2, n_head):
        x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head, mask_enabled=True)
        x = x + ffn(layer_norm(x, **ln_2), **mlp, act_fn=gelu)
        return x

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def __call__(self, inputs):
        x = self.params["wte"][inputs] + self.params["wpe"][range(len(inputs))]
        for block in self.params["blocks"]:
            x = self.transformer_block(x, **block, n_head=self.n_head)
        if self.params["lm_head"]:
            return layer_norm(x, **self.params["ln_f"]) @ self.params["lm_head"].T
        return layer_norm(x, **self.params["ln_f"]) @ self.params["wte"].T

    def generate(self, context, n_tokens=100, topk=5):
        inputs = self.encode(context)
        for _ in range(n_tokens):
            logits = self(inputs)
            next_id = random.choice(np.argsort(logits[-1])[-topk:])
            inputs.append(int(next_id))
        return self.tokenizer.decode(inputs)
