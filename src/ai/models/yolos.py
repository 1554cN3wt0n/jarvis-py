import os
import pickle
from src.ai.nn import (
    layer_norm,
    ffn,
    mha,
    linear,
    sigmoid,
    convolution_2d,
    relu,
    gelu,
)
from src.ai.features.image_features import resize_bicubic
from src.ai.features.utils import gauss_norm
from PIL import Image
import numpy as np
import json


class Yolos:
    def __init__(self, model_path=None, config_path=None, n_head=3):
        if model_path is None:
            model_path = os.getenv("YOLOS_MODEL_PATH")
        if config_path is None:
            config_path = os.getenv("YOLOS_CONFIG_PATH")
        self.n_head = n_head
        self.load_model(model_path)
        self.load_config(config_path)

    def load_model(self, model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            self.params = model["params"]
            self.hparams = model["hparams"]

    def load_config(self, config_path):
        with open(config_path, "r", encoding="utf8") as f:
            data = json.load(f)
            self.id2label = data["id2label"]

    def transformer_block(self, x, mlp, attn, ln_1, ln_2, n_head):
        x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
        x = x + ffn(layer_norm(x, **ln_2), **mlp, act_fn=gelu)
        return x

    def yolos_interpolation(
        self,
        position_embeddings,
        detection_tokens,
        img_size,
        patch_size=16,
        config_image_size=(800, 1333),
    ):
        num_detection_tokens = detection_tokens.shape[0]
        cls_pos_emb = position_embeddings[:1]
        det_pos_emb = position_embeddings[-num_detection_tokens:]

        patch_pos_emb = position_embeddings[1:-num_detection_tokens]
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

        scale_pos_emb = np.concatenate([cls_pos_emb, patch_pos_emb, det_pos_emb])
        return scale_pos_emb

    def yolos_embeddings(
        self, inputs, cls_token, detection_tokens, position_embeddings, conv_proj
    ):
        x = convolution_2d(
            inputs,
            conv_proj["w"],
            bias=conv_proj["b"],
            stride=16,
        )
        x = x.reshape(x.shape[0], -1).T
        x = np.vstack([cls_token, x, detection_tokens])

        scale_pos_emb = self.yolos_interpolation(
            position_embeddings,
            detection_tokens,
            img_size=(inputs.shape[1], inputs.shape[2]),
        )
        return scale_pos_emb + x

    def __call__(self, inputs):
        x = self.yolos_embeddings(inputs, **self.params["embeddings"])
        for block in self.params["encoder_blocks"]:
            x = self.transformer_block(x, **block, n_head=self.n_head)
        x = layer_norm(x, **self.params["ln_f"])
        classes = x[-100:, :]
        bboxes = x[-100:, :]
        for i, block in enumerate(self.params["clc_blocks"]):
            if i == len(self.params["clc_blocks"]) - 1:
                classes = linear(classes, **block)
            else:
                classes = relu(linear(classes, **block))
        for i, block in enumerate(self.params["bbox_blocks"]):
            if i == len(self.params["bbox_blocks"]) - 1:
                bboxes = linear(bboxes, **block)
            else:
                bboxes = relu(linear(bboxes, **block))
        bboxes = sigmoid(bboxes)
        return classes, bboxes

    def detect_objects(self, image: Image):
        raw_img = (
            np.array(image.getdata())
            .reshape(image.height, image.width, 3)
            .transpose(2, 0, 1)
            .astype(float)
        )
        raw_img = gauss_norm(raw_img / 255)
        classes, boxes = self(raw_img)
        label_idxs = np.argmax(classes, axis=1)
        res = []
        for idx, box in zip(label_idxs, boxes):
            if idx < 91:
                res.append({"label": self.id2label[str(idx)], "box": box.tolist()})
        return res
