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
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, default=None, help="1 file wav noisy")
    parser.add_argument("--output", type=str, default=None, help="đường dẫn output tương ứng --input")
    parser.add_argument("--input_dir", type=str, default=None, help="folder wav noisy")
    parser.add_argument("--output_dir", type=str, default=None, help="folder lưu kết quả tương ứng --input_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.get("infer", {}).get("device", "auto"))
    sample_rate = cfg["data"]["sample_rate"]

    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    print(f"[INFO] loaded checkpoint {args.checkpoint} lên {device}")

    if args.input:
        assert args.output, "cần --output khi dùng --input"
        est = enhance_file(model, args.input, sample_rate, device)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        save_audio(args.output, est, sample_rate)
        print(f"[INFO] đã lưu {args.output}")

    elif args.input_dir:
        assert args.output_dir, "cần --output_dir khi dùng --input_dir"
        os.makedirs(args.output_dir, exist_ok=True)
        files = [f for f in os.listdir(args.input_dir) if f.lower().endswith((".wav", ".flac", ".mp3"))]
        for fname in tqdm(files, desc="infer"):
            in_path = os.path.join(args.input_dir, fname)
            out_path = os.path.join(args.output_dir, os.path.splitext(fname)[0] + ".wav")
            est = enhance_file(model, in_path, sample_rate, device)
            save_audio(out_path, est, sample_rate)
        print(f"[INFO] đã xử lý {len(files)} file, lưu tại {args.output_dir}")

    else:
        raise ValueError("cần truyền --input hoặc --input_dir")


if __name__ == "__main__":
    main()
