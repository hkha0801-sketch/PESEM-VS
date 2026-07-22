import csv
import random
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


class SpeechEnhancementDataset(Dataset):
    """Đọc file CSV có 2 cột: noisy,clean (đường dẫn tuyệt đối hoặc tương đối tới file wav).

    Trong lúc train, mỗi audio được cắt (random crop) hoặc pad về đúng
    `segment_seconds` để có thể batch được. Khi eval/test có thể set
    segment_seconds=None để lấy nguyên full audio (batch_size phải =1).
    """

    def __init__(self, csv_path: str, sample_rate: int = 16000,
                 segment_seconds: float = None, train: bool = True):
        self.entries = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append((row["noisy"], row["clean"]))
        self.sample_rate = sample_rate
        self.segment_len = int(segment_seconds * sample_rate) if segment_seconds else None
        self.train = train

    def __len__(self):
        return len(self.entries)

    def _load(self, path: str) -> torch.Tensor:
        wav_np, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
        wav = torch.from_numpy(wav_np.T)  # (C, T)
        if wav.shape[0] > 1:  # stereo -> mono
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav.squeeze(0)  # (T,)

    def _fix_length(self, wav: torch.Tensor) -> torch.Tensor:
        if self.segment_len is None:
            return wav
        n = wav.shape[0]
        if n >= self.segment_len:
            if self.train:
                start = random.randint(0, n - self.segment_len)
            else:
                start = (n - self.segment_len) // 2
            return wav[start:start + self.segment_len]
        else:
            pad = self.segment_len - n
            return torch.nn.functional.pad(wav, (0, pad))

    def __getitem__(self, idx):
        noisy_path, clean_path = self.entries[idx]
        noisy = self._load(noisy_path)
        clean = self._load(clean_path)

        # đồng bộ độ dài trước khi crop cùng vị trí
        min_len = min(noisy.shape[0], clean.shape[0])
        noisy, clean = noisy[:min_len], clean[:min_len]

        if self.segment_len is not None:
            n = noisy.shape[0]
            if n >= self.segment_len:
                start = random.randint(0, n - self.segment_len) if self.train else (n - self.segment_len) // 2
                noisy = noisy[start:start + self.segment_len]
                clean = clean[start:start + self.segment_len]
            else:
                pad = self.segment_len - n
                noisy = torch.nn.functional.pad(noisy, (0, pad))
                clean = torch.nn.functional.pad(clean, (0, pad))

        return noisy, clean


def collate_fn(batch):
    noisy = torch.stack([b[0] for b in batch], dim=0)
    clean = torch.stack([b[1] for b in batch], dim=0)
    return noisy, clean
