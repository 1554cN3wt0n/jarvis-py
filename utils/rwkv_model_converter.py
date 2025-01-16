import torch
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
args = parser.parse_args()


def load_encoder_hparams_and_params(model_path, device="cpu"):
    n_layers = 12
    model = torch.load(model_path, map_location=device)
    for k in model.keys():
        if ".time" in k:
            model[k] = model[k].squeeze()
        model[k] = model[k].float().numpy()
    # vocab embedding. shape [vocab_size, emb_dim] ex. [50257, 768]
    wte = model[f"emb.weight"]

    ln_0 = {
        "g": model["blocks.0.ln0.weight"],
        "b": model["blocks.0.ln0.bias"],
    }

    blocks = []
    for i in range(n_layers):
        ln_1 = {
            "g": model[f"blocks.{i}.ln1.weight"],
            "b": model[f"blocks.{i}.ln1.bias"],
        }
        attn = {
            "decay": model[f"blocks.{i}.att.time_decay"],
            "bonus": model[f"blocks.{i}.att.time_first"],
            "mix_k": model[f"blocks.{i}.att.time_mix_k"],
            "mix_v": model[f"blocks.{i}.att.time_mix_v"],
            "mix_r": model[f"blocks.{i}.att.time_mix_r"],
            "Wk": model[f"blocks.{i}.att.key.weight"],
            "Wv": model[f"blocks.{i}.att.value.weight"],
            "Wr": model[f"blocks.{i}.att.receptance.weight"],
            "Wout": model[f"blocks.{i}.att.output.weight"],
        }
        ln_2 = {
            "g": model[f"blocks.{i}.ln2.weight"],
            "b": model[f"blocks.{i}.ln2.bias"],
        }
        ffn = {
            "mix_k": model[f"blocks.{i}.ffn.time_mix_k"],
            "mix_r": model[f"blocks.{i}.ffn.time_mix_r"],
            "Wk": model[f"blocks.{i}.ffn.key.weight"],
            "Wr": model[f"blocks.{i}.ffn.receptance.weight"],
            "Wv": model[f"blocks.{i}.ffn.value.weight"],
        }
        block = {"ln_1": ln_1, "attn": attn, "ln_2": ln_2, "ffn": ffn}
        blocks.append(block)
    ln_f = {
        "g": model["ln_out.weight"],
        "b": model["ln_out.bias"],
    }
    lm_head = model["head.weight"]
    params = {
        "wte": wte,
        "ln_0": ln_0,
        "blocks": blocks,
        "ln_f": ln_f,
        "lm_head": lm_head,
    }
    hparams = {}
    hparams["n_layers"] = 12
    hparams["n_head"] = 12
    hparams["n_ctx"] = 1024
    hparams["hidden_dim"] = 768
    return hparams, params


hparams, params = load_encoder_hparams_and_params(args.model_path)
with open(args.output_path, "wb") as f:
    pickle.dump({"hparams": hparams, "params": params}, f)
