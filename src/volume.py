import os
import numpy as np
import soundfile as sf

INPUT_FOLDER = "enhanced_0058"
OUTPUT_FOLDER = "output_wav"

TARGET_DBFS = -16.0

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def normalize_rms(audio, target_dbfs):
    rms = np.sqrt(np.mean(audio ** 2))

    if rms == 0:
        return audio

    current_dbfs = 20 * np.log10(rms)

    gain_db = target_dbfs - current_dbfs

    gain = 10 ** (gain_db / 20)

    normalized_audio = audio * gain

    peak = np.max(np.abs(normalized_audio))

    if peak > 1.0:
        normalized_audio = normalized_audio / peak * 0.999

    return normalized_audio

for filename in os.listdir(INPUT_FOLDER):

    if not filename.lower().endswith(".wav"):
        continue

    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    try:
        audio, sample_rate = sf.read(input_path)

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        normalized_audio = normalize_rms(
            audio,
            TARGET_DBFS
        )

        sf.write(
            output_path,
            normalized_audio,
            sample_rate
        )

        print(f"Đã xử lý: {filename}")

    except Exception as e:
        print(f"Lỗi {filename}: {e}")


print("Hoàn thành!")