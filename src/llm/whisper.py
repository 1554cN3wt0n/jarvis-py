import numpy as np
import pickle
from tokenizers import Tokenizer
import os
from src.llm.utils import layer_norm, ffn, mha, gelu, convolution_1d


class Whisper:
    def __init__(self, model_path=None, tokenizer_path=None, n_head=6):
        if model_path is None:
            model_path = os.getenv("WHISPER_MODEL_PATH")
        if tokenizer_path is None:
            tokenizer_path = os.getenv("WHISPER_TOKENIZER_PATH")
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
        x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
        x = x + ffn(layer_norm(x, **ln_2), **mlp, act_fn=gelu)
        return x

    def decoder_transformer_block(
        self, x, mlp, attn, encoder_attn, ln_1, ln_2, ln_3, n_head, kv_states=None
    ):
        x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head, mask_enabled=True)
        if kv_states is not None:
            x = x + mha(
                layer_norm(x, **ln_2),
                **encoder_attn,
                kv_states=kv_states,
                n_head=n_head
            )
        x = x + ffn(layer_norm(x, **ln_3), **mlp, act_fn=gelu)
        return x

    def encoder(self, audio_features, params, hparams):
        # Convolutional layers
        x = gelu(
            convolution_1d(
                audio_features,
                params["encoder"]["conv1"]["w"],
                params["encoder"]["conv1"]["b"],
                padding=1,
            )
        )
        x = gelu(
            convolution_1d(
                x,
                params["encoder"]["conv2"]["w"],
                params["encoder"]["conv2"]["b"],
                stride=2,
                padding=1,
            )
        )

        # Add positional embeddings
        x = x.T + params["encoder"]["embed_positions"]
        # Transformer layers
        for layer in params["encoder"]["blocks"]:
            x = self.transformer_block(x, **layer, n_head=hparams["n_head"])

        # Final layer norm
        x = layer_norm(x, **params["encoder"]["ln_f"])
        return x

    def decoder(self, encoder_output, input_ids, params, hparams):
        # Embed tokens
        token_embeddings = params["decoder"]["embed_tokens"][input_ids]
        positions = np.arange(token_embeddings.shape[0])
        token_embeddings = (
            token_embeddings + params["decoder"]["embed_positions"][positions]
        )

        # Transformer layers
        x = token_embeddings
        for layer in params["decoder"]["blocks"]:
            x = self.decoder_transformer_block(
                x, **layer, kv_states=encoder_output, n_head=hparams["n_head"]
            )

        # Final layer norm
        x = layer_norm(x, **params["decoder"]["ln_f"])
        return x

    def generate(self, audio_features, n_tokens=100):
        # Encode audio
        encoder_output = self.encoder(audio_features, self.params, self.hparams)

        # Initialize decoder inputs
        input_ids = [50257]  # Start of sequence token
        for _ in range(n_tokens):
            logits = self.decoder(encoder_output, input_ids, self.params, self.hparams)
            next_token = np.argmax(logits[-1] @ self.params["proj_out"]["w"])
            if next_token == 50256:
                break
            input_ids.append(next_token)
        return self.tokenizer.decode(input_ids)
