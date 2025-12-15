# save as /big_storage/ali/big_storage/baguette/export_st_model.py
import os
from collections import OrderedDict
import torch
from sentence_transformers import SentenceTransformer


def get_model_state_dict(ckpt):
    raw_dict: OrderedDict = ckpt["state_dict"]
    new_dict = OrderedDict()
    for key, value in raw_dict.items():
        new_dict[key.removeprefix("student.")] = value
    return new_dict


def main():
    # Please replace it with your checkpoint
    ckpt_path = r"/big_storage/ali/big_storage/baguette/runs/lightning_logs/version_6/checkpoints/step=8672000_val_loss=0.06544640.ckpt"
    export_dir = r"/big_storage/ali/big_storage/baguette/exported_italian_minilm"

    os.makedirs(export_dir, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = get_model_state_dict(ckpt)

    # Base architecture used in your code
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()
    model.save(export_dir)
    print(f"Saved SentenceTransformer model to: {export_dir}")


if __name__ == "__main__":
    main()
