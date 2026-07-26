import os
import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi


CLEAN_DIR = "inputTestClean"
ENHANCED_DIR = "output"
RESULT_FILE = "MetricGAN+_EN.txt"


def calc_volume_diff(clean, enhanced):
    rms_clean = np.sqrt(np.mean(clean ** 2) + 1e-12)
    rms_enhanced = np.sqrt(np.mean(enhanced ** 2) + 1e-12)
    
    db_clean = 20 * np.log10(rms_clean)
    db_enhanced = 20 * np.log10(rms_enhanced)
    
    return db_enhanced - db_clean


def main():
    results = []

    for filename in sorted(os.listdir(CLEAN_DIR)):
        if not filename.endswith(".wav"):
            continue

        clean_path = os.path.join(CLEAN_DIR, filename)
        enhanced_path = os.path.join(ENHANCED_DIR, filename)

        if not os.path.exists(enhanced_path):
            continue

        clean, sr_clean = sf.read(clean_path, dtype="float32")
        enhanced, sr_enhanced = sf.read(enhanced_path, dtype="float32")

        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]

        try:
            pesq_score = pesq(
                16000,
                clean,
                enhanced,
                "wb"
            )

            stoi_score = stoi(
                clean,
                enhanced,
                16000,
                extended=False
            )

            vol_diff = calc_volume_diff(clean, enhanced)

            results.append(
                (filename, pesq_score, stoi_score, vol_diff)
            )

        except Exception as e:
            print(f"Lỗi {filename}: {e}")

    with open(RESULT_FILE, "w") as f:
        f.write(
            f"{'File':<30}"
            f"{'PESQ':>10}"
            f"{'STOI':>10}"
            f"{'VolDiff(dB)':>15}\n"
        )

        f.write("-" * 65 + "\n")

        for filename, pesq_score, stoi_score, vol_diff in results:
            f.write(
                f"{filename:<30}"
                f"{pesq_score:>10.3f}"
                f"{stoi_score:>10.3f}"
                f"{vol_diff:>15.3f}\n"
            )

        avg_pesq = np.mean([x[1] for x in results])
        avg_stoi = np.mean([x[2] for x in results])
        avg_vol_diff = np.mean([x[3] for x in results])

        f.write("-" * 65 + "\n")
        f.write(
            f"{'Average':<30}"
            f"{avg_pesq:>10.3f}"
            f"{avg_stoi:>10.3f}"
            f"{avg_vol_diff:>15.3f}\n"
        )

    print(f"Đã lưu kết quả: {RESULT_FILE}")


if __name__ == "__main__":
    main()