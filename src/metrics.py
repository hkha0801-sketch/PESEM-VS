
import os
import numpy as np
import soundfile as sf
import librosa
from pesq import pesq
from pystoi import stoi

CLEAN_DIR = "inputClean"
ENHANCED_DIR = "ouput"
RESULT_FILE = "Model.txt"
SAMPLE_RATE = 16000


def load_and_resample(path, target_sr):
    audio, sr = sf.read(path, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio


def extract_f0(audio, sr):
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        sr=sr,
        fmin=librosa.note_to_hz("C2"),  
        fmax=librosa.note_to_hz("C7")  
    )
    return f0


def calc_f0_rmse(clean, enhanced, sr):
    f0_clean = extract_f0(clean, sr)
    f0_enh = extract_f0(enhanced, sr)

    min_len = min(len(f0_clean), len(f0_enh))
    f0_clean = f0_clean[:min_len]
    f0_enh = f0_enh[:min_len]

    valid = (~np.isnan(f0_clean)) & (~np.isnan(f0_enh))

    if np.sum(valid) == 0:
        return np.nan

    rmse = np.sqrt(
        np.mean(
            (f0_clean[valid] - f0_enh[valid]) ** 2
        )
    )

    return rmse

def main():

    results = []

    for filename in sorted(os.listdir(CLEAN_DIR)):

        if not filename.endswith(".wav"):
            continue

        clean_path = os.path.join(CLEAN_DIR, filename)
        enhanced_path = os.path.join(ENHANCED_DIR, filename)

        if not os.path.exists(enhanced_path):
            print(f"Missing: {filename}")
            continue

        # Doc va resample ve dung SAMPLE_RATE (khong bo qua file nua)
        clean = load_and_resample(clean_path, SAMPLE_RATE)
        enhanced = load_and_resample(enhanced_path, SAMPLE_RATE)

        # Align length
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]

        try:

            pesq_score = pesq(
                SAMPLE_RATE,
                clean,
                enhanced,
                "wb"
            )

            stoi_score = stoi(
                clean,
                enhanced,
                SAMPLE_RATE,
                extended=False
            )

            f0_rmse = calc_f0_rmse(
                clean,
                enhanced,
                SAMPLE_RATE
            )

            results.append(
                (
                    filename,
                    pesq_score,
                    stoi_score,
                    f0_rmse
                )
            )

            print(f"Done: {filename}")

        except Exception as e:
            print(f"Error {filename}: {e}")

    with open(RESULT_FILE, "w") as f:

        f.write(
            f"{'File':<30}"
            f"{'PESQ':>10}"
            f"{'STOI':>10}"
            f"{'F0-RMSE':>15}\n"
        )

        f.write("-" * 70 + "\n")

        for filename, pesq_score, stoi_score, f0_rmse in results:

            f.write(
                f"{filename:<30}"
                f"{pesq_score:>10.3f}"
                f"{stoi_score:>10.3f}"
                f"{f0_rmse:>15.2f}\n"
            )

        avg_pesq = np.mean([x[1] for x in results])
        avg_stoi = np.mean([x[2] for x in results])
        avg_f0 = np.nanmean([x[3] for x in results])

        f.write("-" * 70 + "\n")

        f.write(
            f"{'Average':<30}"
            f"{avg_pesq:>10.3f}"
            f"{avg_stoi:>10.3f}"
            f"{avg_f0:>15.2f}\n"
        )

    print(f"\nSaved to {RESULT_FILE}")


if __name__ == "__main__":
    main()