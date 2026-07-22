"""Tạo train.csv / val.csv / test.csv từ 2 thư mục noisy/ và clean/ chứa file cùng tên.

Ví dụ:
    python scripts/make_csv.py \
        --noisy_dir /data/noisy --clean_dir /data/clean \
        --out_dir data --val_ratio 0.1 --test_ratio 0.1
"""
import argparse
import csv
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noisy_dir", type=str, required=True)
    parser.add_argument("--clean_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extensions", type=str, default=".wav,.flac")
    args = parser.parse_args()

    exts = tuple(args.extensions.split(","))
    noisy_files = {f for f in os.listdir(args.noisy_dir) if f.lower().endswith(exts)}
    clean_files = {f for f in os.listdir(args.clean_dir) if f.lower().endswith(exts)}
    common = sorted(noisy_files & clean_files)

    missing_in_clean = noisy_files - clean_files
    missing_in_noisy = clean_files - noisy_files
    if missing_in_clean:
        print(f"[WARN] {len(missing_in_clean)} file có trong noisy nhưng không có trong clean (bỏ qua)")
    if missing_in_noisy:
        print(f"[WARN] {len(missing_in_noisy)} file có trong clean nhưng không có trong noisy (bỏ qua)")

    if not common:
        raise ValueError("Không tìm thấy file trùng tên nào giữa noisy_dir và clean_dir")

    random.seed(args.seed)
    random.shuffle(common)

    n = len(common)
    n_val = int(n * args.val_ratio)
    n_test = int(n * args.test_ratio)
    n_train = n - n_val - n_test

    splits = {
        "train": common[:n_train],
        "val": common[n_train:n_train + n_val],
        "test": common[n_train + n_val:],
    }

    os.makedirs(args.out_dir, exist_ok=True)
    for split_name, files in splits.items():
        out_path = os.path.join(args.out_dir, f"{split_name}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["noisy", "clean"])
            for fname in files:
                writer.writerow([
                    os.path.join(args.noisy_dir, fname),
                    os.path.join(args.clean_dir, fname),
                ])
        print(f"[INFO] {out_path}: {len(files)} cặp file")


if __name__ == "__main__":
    main()
