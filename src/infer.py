import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from utils import load_config, get_device, load_checkpoint
from model import build_model


def load_audio(path: str, sample_rate: int) -> torch.Tensor:
    wav_np, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
    wav = torch.from_numpy(wav_np.T)  # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav  # (1, T)


def save_audio(path: str, wav: torch.Tensor, sample_rate: int):
    """wav: (1, T) or (T,) tensor"""
    wav_np = wav.squeeze(0).numpy() if wav.dim() == 2 else wav.numpy()
    sf.write(path, wav_np, sample_rate)


def enhance_file(model, path: str, sample_rate: int, device) -> torch.Tensor:
    wav = load_audio(path, sample_rate).to(device)
    with torch.no_grad():
        est = model(wav)
    return est.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                         help="mặc định: configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="mặc định: lấy từ infer.checkpoint trong config")
    parser.add_argument("--input", type=str, default=None, help="1 file wav noisy")
    parser.add_argument("--output", type=str, default=None, help="đường dẫn output tương ứng --input")
    parser.add_argument("--input_dir", type=str, default=None, help="folder wav noisy")
    parser.add_argument("--output_dir", type=str, default=None, help="folder lưu kết quả tương ứng --input_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    infer_cfg = cfg.get("infer", {})
    device = get_device(infer_cfg.get("device", "auto"))
    sample_rate = cfg["data"]["sample_rate"]

    # Ưu tiên CLI, nếu không truyền thì lấy default trong config.yaml (infer: ...)
    checkpoint = args.checkpoint or infer_cfg.get("checkpoint")
    input_path = args.input or infer_cfg.get("input")
    output_path = args.output or infer_cfg.get("output")
    input_dir = args.input_dir or infer_cfg.get("input_dir")
    output_dir = args.output_dir or infer_cfg.get("output_dir")

    assert checkpoint, "cần --checkpoint hoặc set infer.checkpoint trong config.yaml"

    model = build_model(cfg).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    print(f"[INFO] loaded checkpoint {checkpoint} lên {device}")

    if input_path:
        assert output_path, "cần --output hoặc set infer.output trong config.yaml khi dùng input"
        est = enhance_file(model, input_path, sample_rate, device)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        save_audio(output_path, est, sample_rate)
        print(f"[INFO] đã lưu {output_path}")

    elif input_dir:
        assert output_dir, "cần --output_dir hoặc set infer.output_dir trong config.yaml khi dùng input_dir"
        os.makedirs(output_dir, exist_ok=True)
        files = [f for f in os.listdir(input_dir) if f.lower().endswith((".wav", ".flac", ".mp3"))]
        for fname in tqdm(files, desc="infer"):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".wav")
            est = enhance_file(model, in_path, sample_rate, device)
            save_audio(out_path, est, sample_rate)
        print(f"[INFO] đã xử lý {len(files)} file, lưu tại {output_dir}")

    else:
        raise ValueError(
            "cần truyền --input/--input_dir, hoặc set infer.input/infer.input_dir trong config.yaml"
        )


if __name__ == "__main__":
    main()