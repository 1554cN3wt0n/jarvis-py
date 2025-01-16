import numpy as np
from src.ai.nn import sigmoid, layer_norm, softmax, relu
from src.ai.features.utils import sample_probs
import pickle
from tokenizers import Tokenizer
import os

exp = np.exp


class RWKV4:
    def __init__(self, model_path: str = None, tokenizer_path: str = None):
        if model_path is None:
            model_path = os.getenv("RWKV4_MODEL_PATH")
        if tokenizer_path is None:
            tokenizer_path = os.getenv("RWKV4_TOKENIZER_PATH")
        self.load_model(model_path)
        self.load_tokenizer(tokenizer_path)

    def load_model(self, model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            self.params = model["params"]
            self.hparams = model["hparams"]

    def load_tokenizer(self, tokenizer_path):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def time_mixing(
        self,
        x,
        last_x,
        last_num,
        last_den,
        decay,
        bonus,
        mix_k,
        mix_v,
        mix_r,
        Wk,
        Wv,
        Wr,
        Wout,
    ):
        k = Wk @ (x * mix_k + last_x * (1 - mix_k))
        v = Wv @ (x * mix_v + last_x * (1 - mix_v))
        r = Wr @ (x * mix_r + last_x * (1 - mix_r))

        wkv = (last_num + exp(bonus + k) * v) / (last_den + exp(bonus + k))
        rwkv = sigmoid(r) * wkv

        num = exp(-exp(decay)) * last_num + exp(k) * v
        den = exp(-exp(decay)) * last_den + exp(k)

        return Wout @ rwkv, (x, num, den)

    def channel_mixing(self, x, last_x, mix_k, mix_r, Wk, Wr, Wv):
        k = Wk @ (x * mix_k + last_x * (1 - mix_k))
        r = Wr @ (x * mix_r + last_x * (1 - mix_r))
        vk = Wv @ relu(k) ** 2
        return sigmoid(r) * vk, x

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def __call__(self, token, state):
        x = self.params["wte"][token]
        x = layer_norm(x, **self.params["ln_0"])
        for i, block in enumerate(self.params["blocks"]):
            x_ = layer_norm(x, **block["ln_1"])
            dx, state[i][:3] = self.time_mixing(x_, *state[i][:3], **block["attn"])
            x = x + dx

            x_ = layer_norm(x, **block["ln_2"])
            dx, state[i][3] = self.channel_mixing(x_, state[i][3], **block["ffn"])
            x = x + dx

        x = layer_norm(x, **self.params["ln_f"])
        x = self.params["lm_head"] @ x
        probs = softmax(x)

        return probs, state

    def generate(self, context):
        new_tokens = []
        state = np.zeros(
            (self.hparams["n_layers"], 4, self.hparams["hidden_dim"]), dtype=np.float32
        )
        for token in self.encode(context):
            probs, state = self(token, state)

        np.random.seed(0)
        for _ in range(100):
            token = sample_probs(probs)
            new_tokens.append(token)
            probs, state = self(token, state)
        return self.tokenizer.decode(new_tokens)
