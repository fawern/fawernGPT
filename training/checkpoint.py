import os
import torch

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict
):
    os.makedirs(path, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": vars(config),
    }
    torch.save(ckpt, os.path.join(path, f"step_{step}.pt"))

def load_latest(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None
):
    if not os.path.isdir(path):
        return 0

    files = [f for f in os.listdir(path) if f.endswith(".pt")]
    if not files:
        return 0

    latest = sorted(files, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
    blob = torch.load(os.path.join(path, latest), map_location="cpu")
    model.load_state_dict(blob["model"])

    if optimizer is not None and "optimizer" in blob:
        optimizer.load_state_dict(blob["optimizer"])
        
    return int(blob.get("step", 0))
