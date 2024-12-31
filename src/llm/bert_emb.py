import numpy as np
from tokenizers import Tokenizer
import pickle
from src.llm.utils import (
    layer_norm,
    ffn,
    mha,
    gelu,
)
import os


def mean_pooling_and_normalization(x):
    o = np.mean(x, axis=0)
    return o / np.linalg.norm(o)


class BertEmbedding:
    def __init__(self, model_path=None, tokenizer_path=None, n_head=12):
        if model_path is None:
            model_path = os.getenv("BERT_EMB_MODEL_PATH")
        if tokenizer_path is None:
            tokenizer_path = os.getenv("BERT_EMB_TOKENIZER_PATH")
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
        self.tokenizer.no_padding()

    def transformer_block(self, x, mlp, attn, ln_1, ln_2, n_head):
        x = layer_norm(x + mha(x, **attn, n_head=n_head), **ln_1)
        x = layer_norm(x + ffn(x, **mlp, act_fn=gelu), **ln_2)
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
        return x

    def embed(self, text):
        sentence_ids = self.encode(text)
        output = self(sentence_ids, [0] * len(sentence_ids))
        return mean_pooling_and_normalization(output)
