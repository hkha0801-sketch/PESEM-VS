# Speech Enhancement with FullSubNet

A PyTorch-based Speech Enhancement project for training and inference on a local paired noisy-clean speech dataset.

The project implements a **FullSubNet-based speech enhancement pipeline** operating in the time-frequency domain. The model takes a noisy speech waveform as input and predicts a **complex Ideal Ratio Mask (cIRM)** to recover an enhanced speech signal.

---

## 1. Project Overview

The main goal of this project is to remove background noise from noisy speech while preserving the original speech characteristics.

The overall system consists of the following stages:

```text
Noisy Speech
      │
      ▼
Audio Loading & Resampling
      │
      ▼
STFT
      │
      ▼
Magnitude Spectrogram
      │
      ▼
FullSubNet
 ┌─────────────────────────────┐
 │ Frequency-band Model (FB)   │
 │             ↓               │
 │ Sub-band Model (SB)         │
 │             ↓               │
 │ Compressed cIRM             │
 └─────────────────────────────┘
      │
      ▼
Decompress cIRM
      │
      ▼
Complex Mask Application
      │
      ▼
Enhanced Complex Spectrogram
      │
      ▼
iSTFT
      │
      ▼
Enhanced Speech
```

The system can process either:

* A single noisy audio file
* A complete directory containing multiple noisy audio files

---

# 2. Project Structure

```text
speech-enhancement/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── utils.py
│   ├── train.py
│   └── infer.py
│
├── checkpoints/
│   └── best_model_FullSubNet_EN.tar
│
├── input/
│   └── noisy_audio.wav
│
├── output/
│   └── enhanced/
│
├── scripts/
│   └── make_csv.py
│
├── requirements.txt
└── README.md
```

### Main components

| File / Directory      | Description                                                     |
| --------------------- | --------------------------------------------------------------- |
| `configs/config.yaml` | Configuration for dataset, STFT, model, training, and inference |
| `data/`               | Training, validation, and testing CSV files                     |
| `src/dataset.py`      | Loads paired noisy-clean audio samples                          |
| `src/model.py`        | FullSubNet model architecture                                   |
| `src/losses.py`       | Training loss functions                                         |
| `src/metrics.py`      | PESQ, STOI, and SI-SDR evaluation metrics                       |
| `src/utils.py`        | STFT, iSTFT, checkpoint loading, device utilities               |
| `src/train.py`        | Model training script                                           |
| `src/infer.py`        | Speech enhancement inference script                             |
| `checkpoints/`        | Saved model checkpoints                                         |
| `input/`              | Input noisy speech files                                        |
| `output/`             | Enhanced speech results                                         |

---

# 3. Installation

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 4. Dataset Preparation

The training pipeline uses paired:

* **Noisy speech**
* **Clean speech**

Each noisy speech sample should correspond to a clean speech sample.

Example:

```text
noisy/
├── speaker01_001.wav
├── speaker01_002.wav
└── speaker02_001.wav

clean/
├── speaker01_001.wav
├── speaker01_002.wav
└── speaker02_001.wav
```

The filenames should match between the noisy and clean directories.

---

## 4.1 CSV Format

The dataset is described using CSV files:

```text
data/
├── train.csv
├── val.csv
└── test.csv
```

Each CSV contains two columns:

```csv
noisy,clean
/path/to/noisy/file1.wav,/path/to/clean/file1.wav
/path/to/noisy/file2.wav,/path/to/clean/file2.wav
```

The columns are:

* `noisy`: Path to the noisy speech file
* `clean`: Path to the corresponding clean speech file

---

## 4.2 Generate Dataset CSV

If you have two directories containing noisy and clean audio files with matching filenames, run:

```bash
python scripts/make_csv.py \
  --noisy_dir /path/to/noisy \
  --clean_dir /path/to/clean \
  --out_dir data \
  --val_ratio 0.1 \
  --test_ratio 0.1
```

This generates:

```text
data/train.csv
data/val.csv
data/test.csv
```

---

# 5. Configuration

The main configuration file is:

```text
configs/config.yaml
```

Example FullSubNet configuration:

```yaml
data:
  train_csv: "data/train.csv"
  val_csv: "data/val.csv"
  test_csv: "data/test.csv"
  sample_rate: 16000
  segment_seconds: 4.0

stft:
  n_fft: 512
  hop_length: 128
  win_length: 512

model:
  name: "FullSubNet"

  num_freqs: 257
  look_ahead: 2

  sequence_model: "LSTM"

  fb_num_neighbors: 0
  sb_num_neighbors: 15

  fb_output_activate_function: null
  sb_output_activate_function: null

  fb_model_hidden_size: 512
  sb_model_hidden_size: 384

  norm_type: "offline_laplace_norm"
  num_groups_in_drop_band: 2
  weight_init: false
```

The model uses:

```text
Sample Rate = 16 kHz
FFT Size = 512
Hop Length = 128
Window Length = 512
Frequency Bins = 257
```

---

# 6. Training Pipeline

The training pipeline is:

```text
Clean Speech + Noise
        │
        ▼
Generate Noisy Speech
        │
        ▼
Create Noisy-Clean Pairs
        │
        ▼
Create train.csv / val.csv / test.csv
        │
        ▼
Load Audio
        │
        ▼
Resample to 16 kHz
        │
        ▼
Segment / Pad Audio
        │
        ▼
STFT
        │
        ▼
FullSubNet
        │
        ▼
Predict cIRM
        │
        ▼
Calculate Enhancement Loss
        │
        ▼
Backpropagation
        │
        ▼
Update Model
        │
        ▼
Validation
        │
        ▼
Save Best Checkpoint
```

Run training:

```bash
python src/train.py \
  --config configs/config.yaml
```

The best model is saved in:

```text
checkpoints/best.pt
```

The latest checkpoint is saved in:

```text
checkpoints/last.pt
```

To resume training:

```bash
python src/train.py \
  --config configs/config.yaml \
  --resume checkpoints/last.pt
```

---

# 7. Inference Pipeline

The inference pipeline is the most important part of the speech enhancement system.

The complete inference process is:

```text
                    INPUT
                      │
                      ▼
              Noisy WAV Audio
                      │
                      ▼
             Load Audio File
                      │
                      ▼
             Convert to Mono
                      │
                      ▼
            Resample to 16 kHz
                      │
                      ▼
                   STFT
                      │
                      ▼
       Complex Spectrogram X(f,t)
                      │
             ┌────────┴────────┐
             │                 │
             ▼                 ▼
        Magnitude           Phase
          |X|                ∠X
             │                 │
             ▼                 │
        FullSubNet             │
             │                 │
             ▼                 │
     Frequency-Band Model      │
             │                 │
             ▼                 │
       Sub-Band Model          │
             │                 │
             ▼                 │
       Compressed cIRM         │
             │                 │
             ▼                 │
      Decompress cIRM          │
             │                 │
             ▼                 │
     Complex Ratio Mask M      │
             │                 │
             └────────┬────────┘
                      ▼
            Complex Masking
                      │
                      ▼
       Enhanced Spectrogram
                      │
                      ▼
                    iSTFT
                      │
                      ▼
             Enhanced WAV Audio
                      │
                      ▼
                    OUTPUT
```

---

# 8. Detailed FullSubNet Inference Pipeline

## Step 1 — Load Noisy Audio

The inference script reads a noisy WAV file:

```text
input/noisy_audio.wav
```

The audio is loaded as a PyTorch tensor.

If the input is stereo, it is converted to mono:

```text
Stereo Audio
     │
     ▼
Average Left + Right Channels
     │
     ▼
Mono Audio
```

---

## Step 2 — Resample Audio

The input audio is resampled to the configured sample rate:

```text
Input Audio
     │
     ▼
16,000 Hz
```

The model is configured to operate at:

```yaml
sample_rate: 16000
```

---

## Step 3 — STFT

The waveform is transformed from the time domain into the time-frequency domain.

The configuration is:

```yaml
n_fft: 512
hop_length: 128
win_length: 512
```

The STFT produces a complex spectrogram:

```text
X(f,t) = Real(f,t) + j Imag(f,t)
```

From the complex spectrogram, the magnitude is calculated:

```text
|X(f,t)|
```

The magnitude is provided to FullSubNet.

The original complex spectrogram is preserved because its phase information is required to reconstruct the enhanced waveform.

---

## Step 4 — FullSubNet Processing

FullSubNet contains two main stages:

```text
Input Magnitude Spectrogram
          │
          ▼
Frequency-Band Model
          │
          ▼
Frequency-Band Features
          │
          ▼
Sub-Band Model
          │
          ▼
Complex Ratio Mask
```

### Frequency-Band Model

The Frequency-Band Model processes information across the entire frequency axis.

In this project:

```text
Input Size: 257
Hidden Size: 512
Layers: 2
Model: LSTM
```

The model learns global frequency relationships.

---

### Sub-Band Model

The Sub-Band Model focuses on local frequency information.

The configuration uses:

```text
SB neighbors = 15
```

Therefore, the model considers neighboring frequency bins around each target frequency.

The Sub-Band Model predicts:

```text
2 outputs
```

representing:

```text
Real part of cIRM
Imaginary part of cIRM
```

---

## Step 5 — Decompress cIRM

The model predicts a compressed Complex Ideal Ratio Mask:

```text
Compressed cIRM
       │
       ▼
Decompress
       │
       ▼
Complex Ratio Mask
```

The mask contains:

```text
M_real
M_imag
```

---

## Step 6 — Apply Complex Mask

The predicted complex mask is applied to the noisy complex spectrogram.

Given:

```text
X = X_real + jX_imag
```

and:

```text
M = M_real + jM_imag
```

the enhanced spectrogram is:

```text
Y = X × M
```

The real and imaginary components are calculated as:

```text
Y_real = X_real × M_real - X_imag × M_imag

Y_imag = X_real × M_imag + X_imag × M_real
```

The enhanced complex spectrogram is:

```text
Y = Y_real + jY_imag
```

---

## Step 7 — iSTFT

The enhanced complex spectrogram is converted back into a waveform:

```text
Enhanced Complex Spectrogram
          │
          ▼
         iSTFT
          │
          ▼
Enhanced Speech Waveform
```

The output waveform has the same target sample rate:

```text
16 kHz
```

---

# 9. Inference — Single Audio File

To enhance one noisy audio file:

```bash
python src/infer.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model_FullSubNet_EN.tar \
  --input input/noisy_audio.wav \
  --output output/enhanced/noisy_audio_enhanced.wav
```

The result will be saved as:

```text
output/enhanced/noisy_audio_enhanced.wav
```

---

# 10. Inference — Multiple Audio Files

To enhance an entire directory:

```bash
python src/infer.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model_FullSubNet_EN.tar \
  --input_dir input \
  --output_dir output/enhanced
```

Example:

```text
input/
├── noisy_001.wav
├── noisy_002.wav
├── noisy_003.wav
└── noisy_004.wav
```

After inference:

```text
output/
└── enhanced/
    ├── noisy_001.wav
    ├── noisy_002.wav
    ├── noisy_003.wav
    └── noisy_004.wav
```

The output filenames correspond to the input filenames.

---

# 11. Inference Using Paths from config.yaml

Instead of passing `--input_dir` and `--output_dir` through the command line, they can be specified in `configs/config.yaml`:

```yaml
infer:
  device: "auto"
  checkpoint: "checkpoints/best_model_FullSubNet_EN.tar"

  input: null
  output: null

  input_dir: "data/NOISE SPEECH/TEST"
  output_dir: "outputs/enhanced_test"
```

Then run:

```bash
python src/infer.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model_FullSubNet_EN.tar
```

The system will automatically:

```text
data/NOISE SPEECH/TEST
          │
          ▼
      Load WAV Files
          │
          ▼
       FullSubNet
          │
          ▼
    Speech Enhancement
          │
          ▼
outputs/enhanced_test
```

---

# 12. Checkpoint Compatibility

The checkpoint must match the architecture used to create it.

For example, a FullSubNet checkpoint contains parameters such as:

```text
fb_model.sequence_model.*
fb_model.fc_output_layer.*
sb_model.sequence_model.*
sb_model.fc_output_layer.*
```

Therefore, the configuration must use:

```yaml
model:
  name: "FullSubNet"
```

A FullSubNet checkpoint cannot be directly loaded into a CRN model.

For example, these architectures are incompatible:

```text
FullSubNet Checkpoint
        │
        X
        │
        ▼
CRN Model
```

The correct configuration is:

```text
FullSubNet Checkpoint
        │
        ▼
FullSubNet Model
        │
        ▼
Inference
```

---

# 13. Evaluation Metrics

The project supports common speech enhancement evaluation metrics:

* PESQ
* STOI
* SI-SDR

The general evaluation pipeline is:

```text
Noisy Speech
      │
      ▼
FullSubNet
      │
      ▼
Enhanced Speech
      │
      ├───────────────┐
      │               │
      ▼               ▼
Clean Speech      Enhanced Speech
      │               │
      └───────┬───────┘
              ▼
       Evaluation Metrics
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
     PESQ   STOI   SI-SDR
```

Higher values generally indicate better enhancement quality, although each metric has its own interpretation and range.

---

# 14. Recommended Inference Workflow

For a new speech enhancement experiment, the recommended workflow is:

```text
1. Prepare Clean Speech
          │
          ▼
2. Prepare Noise Dataset
          │
          ▼
3. Generate Noisy Speech
          │
          ▼
4. Create train / validation / test splits
          │
          ▼
5. Generate CSV files
          │
          ▼
6. Configure config.yaml
          │
          ▼
7. Train FullSubNet
          │
          ▼
8. Save Best Checkpoint
          │
          ▼
9. Prepare Noisy Test Audio
          │
          ▼
10. Run Inference
          │
          ▼
11. Generate Enhanced Audio
          │
          ▼
12. Compare Noisy vs Enhanced vs Clean
          │
          ▼
13. Calculate PESQ / STOI / SI-SDR
```

---

# 15. Hardware

The project supports:

* NVIDIA CUDA GPU
* Apple Silicon MPS
* CPU

The device can be configured using:

```yaml
train:
  device: "cuda"
```

or:

```yaml
infer:
  device: "auto"
```

Recommended:

```yaml
infer:
  device: "auto"
```

The system automatically selects the available device when supported.

---

# 16. Summary

This project implements a complete speech enhancement pipeline based on FullSubNet:

```text
Noisy Audio
    │
    ▼
Preprocessing
    │
    ▼
STFT
    │
    ▼
Magnitude Spectrogram
    │
    ▼
FullSubNet
    │
    ├── Frequency-Band Model
    │
    └── Sub-Band Model
            │
            ▼
        cIRM Prediction
            │
            ▼
      Complex Masking
            │
            ▼
 Enhanced Complex Spectrogram
            │
            ▼
           iSTFT
            │
            ▼
    Enhanced Speech Audio
```

The main inference command is:

```bash
python src/infer.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best_model_FullSubNet_EN.tar \
  --input_dir input \
  --output_dir output/enhanced
```

This pipeline allows the model to process noisy speech files and generate enhanced speech while preserving the complex time-frequency information required for high-quality waveform reconstruction.
