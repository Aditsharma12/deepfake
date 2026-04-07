import torch
from model.cvit import CViT


def load_cvit(weight_path, net="cvit2", fp16=False):
    device = torch.device("cpu")

    model = CViT()
    checkpoint = torch.load(weight_path, map_location=device)

    # handle different checkpoint formats
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()

    return model