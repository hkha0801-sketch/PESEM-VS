import os
import random
import yaml
import numpy as np
import torch


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda not available, falling back to cpu")
        return torch.device("cpu")
    return torch.device(device_str)


def stft(waveform: torch.Tensor, n_fft: int, hop_length: int, win_length: int) -> torch.Tensor:
    """waveform: (B, T) -> complex spectrogram (B, F, T')"""
    window = torch.hann_window(win_length, device=waveform.device)
    return torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )


def istft(spec: torch.Tensor, n_fft: int, hop_length: int, win_length: int, length: int = None) -> torch.Tensor:
    """spec: complex (B, F, T') -> waveform (B, T)"""
    window = torch.hann_window(win_length, device=spec.device)
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length,
    )


def save_checkpoint(path: str, model, optimizer, epoch: int, best_val: float, cfg: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "best_val": best_val,
            "config": cfg,
        },
        path,
    )


def load_checkpoint(checkpoint_path, model, map_location="cpu"):
    ckpt = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False
    )

    model.load_state_dict(ckpt["model"])

    return ckpt