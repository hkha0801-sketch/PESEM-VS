import numpy as np
import torch

try:
    from pesq import pesq as pesq_fn
except ImportError:
    pesq_fn = None

try:
    from pystoi import stoi as stoi_fn
except ImportError:
    stoi_fn = None


def compute_si_sdr(est: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    est = est - np.mean(est)
    ref = ref - np.mean(ref)
    s_target = np.sum(est * ref) * ref / (np.sum(ref ** 2) + eps)
    e_noise = est - s_target
    return 10 * np.log10((np.sum(s_target ** 2) + eps) / (np.sum(e_noise ** 2) + eps))


def compute_pesq(est: np.ndarray, ref: np.ndarray, sr: int = 16000) -> float:
    if pesq_fn is None:
        raise ImportError("pip install pesq")
    mode = "wb" if sr == 16000 else "nb"
    try:
        return pesq_fn(sr, ref, est, mode)
    except Exception as e:
        print(f"[WARN] PESQ failed: {e}")
        return float("nan")


def compute_stoi(est: np.ndarray, ref: np.ndarray, sr: int = 16000) -> float:
    if stoi_fn is None:
        raise ImportError("pip install pystoi")
    return stoi_fn(ref, est, sr, extended=False)


def evaluate_batch(est: torch.Tensor, ref: torch.Tensor, sr: int = 16000) -> dict:
    """est, ref: (B, T) tensors on CPU. Trả về dict metric trung bình trên batch."""
    est_np = est.detach().cpu().numpy()
    ref_np = ref.detach().cpu().numpy()

    si_sdrs, pesqs, stois = [], [], []
    for e, r in zip(est_np, ref_np):
        si_sdrs.append(compute_si_sdr(e, r))
        if pesq_fn is not None:
            pesqs.append(compute_pesq(e, r, sr))
        if stoi_fn is not None:
            stois.append(compute_stoi(e, r, sr))

    result = {"si_sdr": float(np.mean(si_sdrs))}
    if pesqs:
        result["pesq"] = float(np.nanmean(pesqs))
    if stois:
        result["stoi"] = float(np.mean(stois))
    return result
