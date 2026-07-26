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


def _strip_prefix(state_dict: dict) -> dict:
    """Bỏ tiền tố 'module.' (DataParallel) hoặc 'model.' nếu có."""
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith("module."):
            name = name[len("module."):]
        if name.startswith("model."):
            name = name[len("model."):]
        new_state_dict[name] = v
    return new_state_dict


def load_checkpoint(checkpoint_path, model, optimizer=None, map_location=None):
    """
    Nạp checkpoint vào `model` (và `optimizer` nếu có).

    Hỗ trợ NHIỀU định dạng checkpoint khác nhau:
      1) Checkpoint do chính save_checkpoint() ở trên tạo ra:
         {"epoch", "model_state", "optimizer_state", "best_val", "config"}
      2) Checkpoint dạng thông dụng khác:
         {"model" | "state_dict" | "model_state_dict": <state_dict>}
      3) Checkpoint dạng GAN (generator/discriminator lưu tách riêng),
         ví dụ file MetricGAN+ tham chiếu (epoch, stats, generator,
         discriminator, g_optimizer, d_optimizer):
         - Nếu model có submodule tên "generator" (như
           MetricGANPlusWrapper.generator), state_dict trong
           checkpoint["generator"] sẽ được nạp thẳng vào
           model.generator, KHÔNG bị bỏ qua âm thầm như trước.
      4) Checkpoint là state_dict thuần (không bọc trong dict khác).

    Trả về (start_epoch, best_val) để tương thích với train.py.
    Nếu checkpoint không có epoch/best_val (vd. checkpoint GAN tham
    chiếu từ nguồn khác), mặc định start_epoch=0, best_val=inf.

    QUAN TRỌNG: hàm sẽ RAISE lỗi rõ ràng nếu không nạp được bất kỳ
    trọng số nào, thay vì âm thầm báo "thành công" như bản cũ.
    """

    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    start_epoch = 0
    best_val = float("inf")

    loaded_any = False

    if isinstance(ckpt, dict) and "generator" in ckpt and hasattr(model, "generator"):
        # ---- Checkpoint dạng GAN (MetricGAN+, v.v.) ----
        gen_state = _strip_prefix(ckpt["generator"])
        missing, unexpected = model.generator.load_state_dict(gen_state, strict=False)

        if len(missing) == 0 and len(unexpected) == 0:
            print(f"[INFO] Đã nạp generator state_dict khớp 100% từ {checkpoint_path}")
        else:
            print(f"[WARN] generator load_state_dict: missing={missing}, unexpected={unexpected}")

        loaded_any = len(gen_state) > 0 and len(missing) < len(gen_state)

        if "discriminator" in ckpt and hasattr(model, "discriminator"):
            disc_state = _strip_prefix(ckpt["discriminator"])
            model.discriminator.load_state_dict(disc_state, strict=False)

        start_epoch = int(ckpt.get("epoch", 0))
        best_val = ckpt.get("best_val", ckpt.get("stats", {}).get("pesq", float("inf")))

    elif isinstance(ckpt, dict) and (
        "model" in ckpt or "state_dict" in ckpt
        or "model_state_dict" in ckpt or "model_state" in ckpt
    ):
        # ---- Checkpoint thông dụng: model đơn (FullSubNet, InterSubNet, CRN, ...) ----
        key = next(
            k for k in ("model", "state_dict", "model_state_dict", "model_state")
            if k in ckpt
        )
        state_dict = _strip_prefix(ckpt[key])

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if len(missing) == 0 and len(unexpected) == 0:
            print(f"[INFO] Đã nạp state_dict khớp 100% từ {checkpoint_path}")
        else:
            print(f"[WARN] load_state_dict: missing={missing}, unexpected={unexpected}")

        loaded_any = len(state_dict) > 0 and len(missing) < len(state_dict)

        start_epoch = int(ckpt.get("epoch", 0))
        best_val = ckpt.get("best_val", float("inf"))

        if optimizer is not None and ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])

    else:
        # ---- Checkpoint là state_dict thuần ----
        state_dict = _strip_prefix(ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        loaded_any = len(state_dict) > 0 and len(missing) < len(state_dict)

    if not loaded_any:
        raise RuntimeError(
            f"[load_checkpoint] KHÔNG nạp được bất kỳ trọng số nào từ "
            f"'{checkpoint_path}'. Kiểm tra lại cấu trúc checkpoint và "
            f"kiến trúc model có khớp nhau không (xem log missing/unexpected ở trên)."
        )

    print(f"[INFO] Successfully loaded checkpoint from {checkpoint_path}")

    return start_epoch, best_val