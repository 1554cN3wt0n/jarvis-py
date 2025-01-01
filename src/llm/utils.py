import numpy as np


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps=1e-12):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b


def linear(x, w, b):
    return x @ w + b


def ffn(x, c_fc, c_proj, act_fn=relu):
    return linear(act_fn(linear(x, **c_fc)), **c_proj)


def attention(q, k, v, mask=None):
    if mask is None:
        return softmax(q @ k.T / np.sqrt(q.shape[-1])) @ v
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head, mask_enabled=False):
    x = linear(x, **c_attn)
    qkv_heads = list(
        map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1))
    )
    causal_mask = None
    if mask_enabled:
        causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, mask=causal_mask) for q, k, v in zip(*qkv_heads)]
    x = linear(np.hstack(out_heads), **c_proj)
    return x
