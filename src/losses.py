import torch
import torch.nn as nn

from utils import stft


def si_snr_loss(est: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-Invariant SNR loss (âm để minimize -> loss càng nhỏ SI-SNR càng cao).
    est, target: (B, T)
    """
    est = est - est.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    s_target = (torch.sum(est * target, dim=-1, keepdim=True) * target /
                (torch.sum(target ** 2, dim=-1, keepdim=True) + eps))
    e_noise = est - s_target

    si_snr = 10 * torch.log10(
        (torch.sum(s_target ** 2, dim=-1) + eps) / (torch.sum(e_noise ** 2, dim=-1) + eps)
    )
    return -si_snr.mean()


def spectral_loss(est: torch.Tensor, target: torch.Tensor, n_fft=512, hop_length=128, win_length=512) -> torch.Tensor:
    """L1 loss trên magnitude spectrogram."""
    est_spec = torch.abs(stft(est, n_fft, hop_length, win_length))
    tgt_spec = torch.abs(stft(target, n_fft, hop_length, win_length))
    return torch.nn.functional.l1_loss(est_spec, tgt_spec)


def get_loss_fn(name: str, stft_cfg: dict = None):
    if name == "sisnr":
        return lambda est, tgt: si_snr_loss(est, tgt)
    elif name == "spectral":
        return lambda est, tgt: spectral_loss(est, tgt, **stft_cfg)
    elif name == "combo":
        def combo(est, tgt):
            return si_snr_loss(est, tgt) + spectral_loss(est, tgt, **stft_cfg)
        return combo
    else:
        raise ValueError(f"Unknown loss: {name}")
