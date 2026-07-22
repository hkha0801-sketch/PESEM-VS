# Speech Enhancement (PyTorch)

Repo huấn luyện & inference model khử nhiễu tiếng nói (Speech Enhancement) từ
dataset local, dùng kiến trúc CRN (Convolutional Recurrent Network) trên miền
STFT (mask-based, dự đoán complex ratio mask).

## 1. Cấu trúc thư mục

```
speech-enhancement/
├── configs/
│   └── config.yaml          # toàn bộ hyperparameter, đường dẫn
├── data/
│   ├── train.csv            # cột: noisy,clean
│   ├── val.csv
│   └── test.csv
├── src/
│   ├── dataset.py           # Dataset + collate_fn (đọc wav, cắt/pad theo segment)
│   ├── model.py             # kiến trúc CRN mask-based
│   ├── losses.py            # loss (SI-SNR, spectral loss)
│   ├── metrics.py           # PESQ, STOI, SI-SDR
│   ├── utils.py             # STFT/ISTFT, load config, seed, checkpoint
│   ├── train.py             # script train
│   └── infer.py             # script inference (1 file hoặc cả folder)
├── checkpoints/              # nơi lưu model .pt
├── scripts/
│   └── make_csv.py          # tool tạo train/val/test.csv từ 2 thư mục noisy/clean
├── requirements.txt
└── README.md
```

## 2. Cài đặt

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 3. Chuẩn bị dataset

Bạn cần dataset gồm cặp file **noisy** (tiếng ồn) và **clean** (tiếng sạch),
cùng sample rate, cùng độ dài (hoặc không cần cùng độ dài, code sẽ tự cắt/pad).

Format CSV (`data/train.csv`, `data/val.csv`, `data/test.csv`):

```csv
noisy,clean
/path/to/noisy/file1.wav,/path/to/clean/file1.wav
/path/to/noisy/file2.wav,/path/to/clean/file2.wav
```

Nếu bạn có 2 thư mục `noisy/` và `clean/` với file cùng tên, dùng script có sẵn:

```bash
python scripts/make_csv.py \
  --noisy_dir /path/to/noisy \
  --clean_dir /path/to/clean \
  --out_dir data \
  --val_ratio 0.1 --test_ratio 0.1
```

## 4. Train

Chỉnh `configs/config.yaml` (sample_rate, batch_size, epochs, đường dẫn csv...)
rồi chạy:

```bash
python src/train.py --config configs/config.yaml
```

- Checkpoint tốt nhất (theo val loss) lưu ở `checkpoints/best.pt`
- Checkpoint mỗi epoch lưu ở `checkpoints/last.pt`
- Có resume: `python src/train.py --config configs/config.yaml --resume checkpoints/last.pt`
- Log train/val loss in ra console mỗi epoch (có thể pipe ra file hoặc bật TensorBoard, xem `train.py`)

## 5. Inference

Khử nhiễu 1 file:

```bash
python src/infer.py --config configs/config.yaml \
  --checkpoint checkpoints/best.pt \
  --input noisy_sample.wav \
  --output enhanced_sample.wav
```

Khử nhiễu cả folder:

```bash
python src/infer.py --config configs/config.yaml \
  --checkpoint checkpoints/best.pt \
  --input_dir ./test_noisy \
  --output_dir ./test_enhanced
```

## 6. Đánh giá (metrics)

`src/metrics.py` cung cấp PESQ, STOI, SI-SDR. Có thể chạy đánh giá trên tập
test bằng cách gọi trực tiếp trong `train.py` (tự động chạy sau khi train xong
nếu `test.csv` tồn tại) hoặc viết script riêng import các hàm trong file này.

## 7. Đổi kiến trúc model

Model mặc định là CRN mask-based trong `src/model.py` (nhẹ, train nhanh, chất
lượng tốt cho baseline). Nếu muốn nâng cấp lên kiến trúc mạnh hơn (Conv-TasNet,
DCCRN, DEMUCS-denoiser...), chỉ cần viết class model mới cùng interface
`forward(noisy_waveform) -> enhanced_waveform` trong `src/model.py` và cập nhật
tên class trong `configs/config.yaml`.

## 8. Ghi chú phần cứng

- Tự động dùng GPU nếu có (`cuda`), fallback CPU.
- Config có `num_workers` cho DataLoader, chỉnh theo số core CPU.
