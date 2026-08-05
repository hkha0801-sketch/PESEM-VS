import os
import numpy as np
import soundfile as sf



CLEAN_DIR = "CLEAN"
NOISE_DIR = "NOISE"
OUTPUT_DIR = "NOISE SPEECH"

SNR_DB = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

def match_noise_length(noise, target_length):
    """
    Điều chỉnh noise có cùng độ dài với clean.
    Nếu noise ngắn -> lặp lại.
    Nếu noise dài -> cắt.
    """

    if len(noise) < target_length:
        repeat_times = int(np.ceil(target_length / len(noise)))
        noise = np.tile(noise, repeat_times)

    noise = noise[:target_length]

    return noise



def mix_audio(clean, noise, snr_db):

    clean_power = np.mean(clean ** 2)

    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return clean
    
    snr_linear = 10 ** (snr_db / 10)

    noise_scale = np.sqrt(
        clean_power / (snr_linear * noise_power)
    )

    noise = noise * noise_scale

    noisy = clean + noise

    return noisy



clean_files = [
    f for f in os.listdir(CLEAN_DIR)
    if f.lower().endswith(".wav")
]

noise_files = [
    f for f in os.listdir(NOISE_DIR)
    if f.lower().endswith(".wav")
]

clean_files.sort()
noise_files.sort()


print(f"Số file clean: {len(clean_files)}")
print(f"Số file noise: {len(noise_files)}")
print(f"Tổng file noisy sẽ tạo: {len(clean_files) * len(noise_files)}")

count = 0

for clean_file in clean_files:

    clean_path = os.path.join(
        CLEAN_DIR,
        clean_file
    )

    clean, clean_sr = sf.read(clean_path)

    if clean.ndim > 1:
        clean = np.mean(clean, axis=1)


    for noise_file in noise_files:

        noise_path = os.path.join(
            NOISE_DIR,
            noise_file
        )

        noise, noise_sr = sf.read(noise_path)

        if noise.ndim > 1:
            noise = np.mean(noise, axis=1)

        if clean_sr != noise_sr:
            print(
                f"SKIP: {clean_file} + {noise_file} "
                f"(Sample rate khác nhau)"
            )
            continue


        noise = match_noise_length(
            noise,
            len(clean)
        )


        noisy = mix_audio(
            clean,
            noise,
            SNR_DB
        )


        clean_name = os.path.splitext(
            clean_file
        )[0]

        noise_name = os.path.splitext(
            noise_file
        )[0]

        output_name = (
            f"{clean_name}-{noise_name}.wav"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            output_name
        )


        sf.write(
            output_path,
            noisy,
            clean_sr
        )

        count += 1

        print(
            f"[{count}] "
            f"{output_name}"
        )


print("\nHoàn thành!")
print(f"Đã tạo: {count} file noisy")