import os
import pickle
from src.llm.utils import (
    layer_norm,
    ffn,
    mha,
    linear,
    convolution_2d,
    gelu,
)
from src.features.image_features import resize_bicubic
import numpy as np


class ViT:
    def __init__(self, model_path=None, n_head=3):
        if model_path is None:
            model_path = os.getenv("VIT_MODEL_PATH")
        self.n_head = n_head
        self.load_model(model_path)

    def load_model(self, model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            self.params = model["params"]
            self.hparams = model["hparams"]

    def transformer_block(self, x, mlp, attn, ln_1, ln_2, n_head):
        x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
        x = x + ffn(layer_norm(x, **ln_2), **mlp, act_fn=gelu)
        return x

    def vit_interpolation(
        self,
        position_embeddings,
        img_size,
        patch_size=16,
        config_image_size=(224, 224),
    ):
        cls_pos_emb = position_embeddings[:1]

        patch_pos_emb = position_embeddings[1:]
        patch_pos_emb = patch_pos_emb.T
        hidden_size, seq_len = patch_pos_emb.shape

        patch_height, patch_width = (
            config_image_size[0] // patch_size,
            config_image_size[1] // patch_size,
        )
        patch_pos_emb = patch_pos_emb.reshape(hidden_size, patch_height, patch_width)

        height, width = img_size
        new_patch_height, new_patch_width = (
            height // patch_size,
            width // patch_size,
        )

        patch_pos_emb = resize_bicubic(patch_pos_emb, new_patch_height, new_patch_width)

        patch_pos_emb = patch_pos_emb.reshape(hidden_size, -1).transpose(1, 0)

        scale_pos_emb = np.concatenate([cls_pos_emb, patch_pos_emb])
        return scale_pos_emb

    def vit_embeddings(self, inputs, cls_token, position_embeddings, conv_proj):
        x = convolution_2d(
            inputs,
            conv_proj["w"],
            bias=conv_proj["b"],
            stride=16,
        )
        x = x.reshape(x.shape[0], -1).T
        x = np.vstack([cls_token, x])

        scale_pos_emb = self.vit_interpolation(
            position_embeddings,
            img_size=(inputs.shape[1], inputs.shape[2]),
        )
        return scale_pos_emb + x

    def __call__(self, inputs):
        x = self.vit_embeddings(inputs, **self.params["embeddings"])
        for block in self.params["encoder_blocks"]:
            x = self.transformer_block(x, **block, n_head=self.n_head)
        x = layer_norm(x, **self.params["ln_f"])
        logits = linear(x, **self.params["classifier"])
        return logits[0]
