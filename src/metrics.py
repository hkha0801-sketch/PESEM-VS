import os
import librosa
import numpy as np

from pesq import pesq
from pystoi import stoi

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


CLEAN_DIR = os.path.join(ROOT_DIR, "input")
ENHANCED_DIR = os.path.join(ROOT_DIR, "output", "enhanced")
RESULT_FILE = os.path.join(ROOT_DIR, "metrics_result.txt")

SAMPLE_RATE = 16000

pesq_scores = []
stoi_scores = []

result_lines = []

header = f"{'File':35s} {'PESQ':>8} {'STOI':>8}"
print(header)
print("-" * len(header))

result_lines.append(header)
result_lines.append("-" * len(header))

for file in sorted(os.listdir(CLEAN_DIR)):

    if not file.endswith(".wav"):
        continue

    clean_path = os.path.join(CLEAN_DIR, file)
    enhanced_path = os.path.join(ENHANCED_DIR, file)

    if not os.path.exists(enhanced_path):
        continue

    try:

        clean, _ = librosa.load(clean_path, sr=SAMPLE_RATE)
        enhanced, _ = librosa.load(enhanced_path, sr=SAMPLE_RATE)

        length = min(len(clean), len(enhanced))
        clean = clean[:length]
        enhanced = enhanced[:length]

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

        pesq_scores.append(pesq_score)
        stoi_scores.append(stoi_score)

        line = f"{file:35s} {pesq_score:8.3f} {stoi_score:8.3f}"

        print(line)
        result_lines.append(line)

    except:
        continue


print("-" * len(header))
result_lines.append("-" * len(header))

if len(pesq_scores) > 0:

    avg_pesq = np.mean(pesq_scores)
    avg_stoi = np.mean(stoi_scores)

    avg_line = f"{'Average':35s} {avg_pesq:8.3f} {avg_stoi:8.3f}"

    print(avg_line)
    result_lines.append(avg_line)

else:

    print("No valid audio pairs found.")
    result_lines.append("No valid audio pairs found.")


with open(RESULT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(result_lines))

print(f"\nResults saved to: {RESULT_FILE}")