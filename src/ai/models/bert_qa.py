import numpy as np
from tokenizers import Tokenizer
import pickle
from src.ai.nn import (
    layer_norm,
    linear,
    ffn,
    mha,
    relu,
)
import os


class BertQA:
    def __init__(self, model_path=None, tokenizer_path=None, n_head=12):
        if model_path is None:
            model_path = os.getenv("BERT_MODEL_PATH")
        if tokenizer_path is None:
            tokenizer_path = os.getenv("BERT_TOKENIZER_PATH")
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
        x = layer_norm(x + mha(x, **attn, n_head=n_head), **ln_1)
        x = layer_norm(x + ffn(x, **mlp, act_fn=relu), **ln_2)
        return x

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def __call__(self, inputs, segment_ids):

        x = (
            self.params["wte"][inputs]
            + self.params["wpe"][range(len(inputs))]
            + self.params["wtte"][segment_ids]
        )
        x = layer_norm(x, **self.params["ln_0"])
        for block in self.params["blocks"]:
            x = self.transformer_block(x, **block, n_head=self.n_head)
        return linear(x, **self.params["qa"])

    def answer(self, question, context):
        question_ids = self.encode(question)
        context_ids = self.encode(context)
        input_ids = question_ids + context_ids[1:]
        token_type_ids = [0] * len(question_ids) + [1] * len(context_ids[1:])
        logits = self(input_ids, token_type_ids)
        idx0 = np.argmax(logits[:, 0])
        idx1 = np.argmax(logits[:, 1])
        return self.tokenizer.decode(input_ids[idx0 : idx1 + 1])
