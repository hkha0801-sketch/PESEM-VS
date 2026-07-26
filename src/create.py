import os
import shutil

INPUT_DIR = "inputTestClean"
OUTPUT_DIR = "inputTestClean47"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(".wav"):
        continue

    name = os.path.splitext(filename)[0]
    src = os.path.join(INPUT_DIR, filename)

    for i in range(1, 48):
        new_name = f"{name}-NO{i:02d}.wav"
        dst = os.path.join(OUTPUT_DIR, new_name)

        shutil.copy2(src, dst)

print("Đã tạo xong!")