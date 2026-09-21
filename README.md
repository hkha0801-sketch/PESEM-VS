# PESEM-VS

## An Empirical Study on Tonal Fidelity Preservation for Speech Enhancement Model Selection in Vietnamese

**PESEM-VS** is an empirical benchmark for studying how modern speech enhancement architectures preserve **Vietnamese tonal information** under zero-shot cross-lingual evaluation and in-domain fine-tuning.

Most speech enhancement systems are developed and evaluated primarily on English-language corpora. Vietnamese is a tonal language in which fundamental-frequency (**F0**) contours carry lexical information, so perceptual improvement alone may not guarantee preservation of linguistically meaningful pitch patterns.

This repository provides the evaluation code, fine-tuning notebooks, experiment results, and supporting material used in the PESEM-VS study.

---

## Overview

The study addresses two main research questions:

1. **How well do English-pretrained speech enhancement models generalize to Vietnamese speech without adaptation?**
2. **Does in-domain Vietnamese fine-tuning improve both perceptual speech quality and tonal preservation?**

The evaluation uses four complementary metrics:

- **PESQ** — perceptual speech quality
- **STOI** — speech intelligibility
- **F0-RMSE** — fundamental-frequency reconstruction error
- **PFR** — Pitch Failure Rate

---

## Evaluated Speech Enhancement Architectures

| Category | Models |
|---|---|
| Sub-band / Multi-stage | FullSubNet, FullSubNet+, InterSubNet |
| Lightweight / Resource-efficient | SEMamba, GTCRN |
| Attention / Conformer-based | MANNER, CMGAN, MP-SENet |
| Generative / Objective-driven | MetricGAN+, SGMSE |

### Architecture Taxonomy

<p align="center">
  <img src="./assets/taxonomy.png" alt="Taxonomy of speech enhancement paradigms" width="760">
</p>

---

## Dataset

The PESEM-VS evaluation corpus contains:

- **200 clean Vietnamese utterances**
- **47 environmental noise recordings**
- **9,400 noisy-clean speech pairs**
- **10 dB fixed SNR**
- Multiple Vietnamese regional accents
- Both synthetic and human speech sources

### Dataset Split

| Split | Scripts | Clean Speech | Noisy Speech |
|---|---:|---:|---:|
| Train | S03–S20 | 180 | 8,460 |
| Test | S01–S02 | 20 | 940 |
| **Total** | S01–S20 | **200** | **9,400** |

Dataset:

https://huggingface.co/datasets/KhaBui/PESEM-VS

---

## Experimental Pipeline

<p align="center">
  <img src="./assets/pipeline.png" alt="PESEM-VS experimental pipeline" width="900">
</p>

---

## Evaluation Metrics

### PESQ

**Perceptual Evaluation of Speech Quality (PESQ)** measures perceptual speech quality.

Higher is better.

### STOI

**Short-Time Objective Intelligibility (STOI)** measures speech intelligibility.

Higher is better.

### F0-RMSE

To evaluate tonal preservation, fundamental-frequency (**F0**) contours are extracted from both clean and enhanced speech using **pYIN**.

F0-RMSE is calculated over frames that are identified as voiced in both signals:

```math
\mathrm{F0\text{-}RMSE}
=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left(
F_{0}^{\mathrm{clean}}(i)
-
F_{0}^{\mathrm{enh}}(i)
\right)^2
}
```

where \(N\) is the number of valid voiced frames.

Lower values indicate closer preservation of the reference pitch contour.

### Pitch Failure Rate (PFR)

F0-RMSE only evaluates frames for which pitch can be successfully estimated.

**Pitch Failure Rate (PFR)** complements F0-RMSE by measuring the percentage of reference voiced frames for which the enhanced speech fails to produce a valid corresponding F0 estimate.

```math
\mathrm{PFR}
=
\frac{N_{\mathrm{failure}}}{N_{\mathrm{reference\ voiced}}}
\times 100\%
```

Lower values are better. A PFR of **0%** means that a valid F0 estimate is obtained for every reference voiced frame.

---

## Results

### Zero-Shot Cross-Lingual Evaluation

The complete zero-shot results are available in:

```text
results/zero_shot.csv
```

### In-Domain Fine-Tuning

The complete fine-tuned results are available in:

```text
results/finetuned.csv
```

### Checkpoint Trajectory Analysis

MP-SENet and CMGAN are evaluated across **20 fine-tuning checkpoints**.

CMGAN shows rapid PESQ improvement during the earlier stages of training and approaches a narrow PESQ range around the later checkpoints, while its F0-RMSE continues to decrease.

MP-SENet starts from a stronger zero-shot baseline and maintains lower absolute F0-RMSE than CMGAN throughout the evaluated checkpoints.

Its PESQ continues improving until approximately checkpoint 19, with no comparable plateau observed within the evaluated training horizon.

<p align="center">
  <img src="./assets/checkpoint_trajectory.png" alt="Checkpoint trajectories of MP-SENet and CMGAN" width="760">
</p>

The numerical checkpoint results are available in:

```text
results/checkpoint_trajectories.csv
```

---

## Checkpoints

Model checkpoints are stored externally to keep the Git repository lightweight.

**Google Drive:**  
[PESEM-VS Checkpoints](https://drive.google.com/drive/folders/1juuIjFd1qkZ4gYRTwWHAXudvqHP4F89f?usp=drive_link)

---

## Main Findings

### 1. Perceptual quality does not guarantee tonal preservation

Models can achieve acceptable PESQ or STOI while introducing substantial F0 distortion. Pitch-aware evaluation is therefore useful when studying speech enhancement for tonal languages.

### 2. Fine-tuning behavior is architecture-dependent

Vietnamese adaptation improves MP-SENet, CMGAN, SEMamba, and SGMSE across important perceptual and pitch-related dimensions, while the evaluated sub-band models show limited or negative transfer under the same experimental protocol.

### 3. MP-SENet provides the strongest balance in this evaluation

Across the evaluated conditions, MP-SENet provides the strongest overall combination of PESQ, STOI, F0-RMSE, and PFR.

---

## Repository Structure

The repository currently follows this structure:

```text
PESEM-VS/
├── README.md
├── requirements.txt
├── dataset.yaml
├── .gitignore
│
├── assets/
│
├── configs/
│
├── data/
│
├── results/
│   ├── zero_shot.csv
│   ├── finetuned.csv
│   └── checkpoint_trajectories.csv
│
├── scripts/
│   └── [training / fine-tuning notebooks]
│
└── src/
    ├── metrics.py
    └── pfr.py
```

No model-weight files are stored directly in the repository. Checkpoints are provided through the Google Drive link above.

---

## Installation

```bash
git clone https://github.com/hkha0801-sketch/PESEM-VS.git
cd PESEM-VS
pip install -r requirements.txt
```

---

## Notes

The scripts/notebooks use the common evaluation implementation in `src/metrics.py` and `src/pfr.py` for the reported objective metrics.

For model-specific dependencies and pretrained initialization, refer to the corresponding fine-tuning notebook and the original implementation of each architecture.
