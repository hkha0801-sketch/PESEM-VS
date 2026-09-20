# PESEM-VS

## An Empirical Study on Tonal Fidelity Preservation for Speech Enhancement Model Selection in Vietnamese

**PESEM-VS** is an empirical benchmark for studying how modern speech enhancement architectures preserve **Vietnamese tonal information** under zero-shot cross-lingual evaluation and in-domain fine-tuning.

Most speech enhancement systems are developed and evaluated primarily on English-language corpora. However, Vietnamese is a tonal language in which fundamental-frequency (**F0**) contours carry lexical information. A model may therefore improve perceptual speech quality while still distorting linguistically meaningful pitch patterns.

This repository provides the code, evaluation pipeline, experimental results, and supporting material for our study of **10 representative speech enhancement architectures** using both conventional speech-quality metrics and pitch-aware evaluation.

---

## Overview

The study addresses two main research questions:

1. **How well do English-pretrained speech enhancement models generalize to Vietnamese speech without adaptation?**
2. **Does in-domain Vietnamese fine-tuning improve both perceptual speech quality and tonal preservation?**

To answer these questions, we evaluate the models using four complementary metrics:

- **PESQ** — perceptual speech quality
- **STOI** — speech intelligibility
- **F0-RMSE** — fundamental-frequency reconstruction error
- **PFR** — Pitch Failure Rate

The study evaluates zero-shot performance, Vietnamese fine-tuning behavior, multi-criteria model selection, and checkpoint-level training trajectories.

---

## Key Contributions

- A systematic evaluation of **10 speech enhancement architectures** on Vietnamese speech.
- A unified evaluation protocol combining **PESQ, STOI, F0-RMSE, and PFR**.
- Analysis of how architecture design affects tonal preservation before and after Vietnamese fine-tuning.
- Multi-criteria comparison of perceptual quality, intelligibility, and pitch preservation.
- Checkpoint-level analysis of the leading architectures during fine-tuning.

---

## Speech Enhancement Architectures

The evaluated models cover four representative architectural families.

| Category | Models |
|---|---|
| Sub-band / Multi-stage | FullSubNet, FullSubNet+, InterSubNet |
| Lightweight / Resource-efficient | SEMamba, GTCRN |
| Attention / Conformer-based | MANNER, CMGAN, MP-SENet |
| Generative / Objective-driven | MetricGAN+, SGMSE |

### Evaluated Models

| Model | Architecture Family | Reference |
|---|---|---|
| FullSubNet | Sub-band | ICASSP 2021 |
| FullSubNet+ | Sub-band | ICASSP 2022 |
| InterSubNet | Sub-band | ICASSP 2023 |
| SEMamba | State-space / Lightweight | SLT 2024 |
| GTCRN | Lightweight | ICASSP 2024 |
| MANNER | Attention | ICASSP 2022 |
| CMGAN | Conformer / GAN | Interspeech 2022 |
| MP-SENet | Conformer / Phase-aware | Interspeech 2023 |
| MetricGAN+ | Metric-driven GAN | Interspeech 2021 |
| SGMSE | Score-based Generative | IEEE/ACM TASLP 2023 |

The broader model-selection stage also considered Conv-TasNet, DeepFilterNet, Spiking-FullSubNet, LiSenNet, and FSPEN. These models were excluded from the final experimental set according to task compatibility, sampling-rate compatibility, or architectural redundancy with respect to the study's tonal-preservation analysis.

---

## Dataset

The PESEM-VS evaluation corpus contains:

- **200 clean Vietnamese utterances**
- **47 environmental noise recordings**
- **9,400 noisy-clean speech pairs**
- **10 dB fixed SNR**
- Multiple Vietnamese regional accents
- Both synthetic and human speech sources

Each of the 200 clean utterances is mixed with all 47 noise recordings:

\[
200 \times 47 = 9,400
\]

### Clean Speech Composition

The clean corpus is constructed from:

- 20 script sentences
- 2 voice-source types
  - AI-generated speech
  - Human-recorded speech
- 5 regional accent groups per source

The scripts cover several domains, including:

- customer-service speech
- weather forecasts
- contact information
- code-switched content

### Dataset Split

The corpus is split by **script identity** rather than by individual noisy mixtures.

| Split | Scripts | Clean Speech | Noisy Speech |
|---|---:|---:|---:|
| Train | S03–S20 | 180 | 8,460 |
| Test | S01–S02 | 20 | 940 |
| **Total** | S01–S20 | **200** | **9,400** |

Accent and voice-source distributions are kept consistent across the training and test partitions.

Noise conditions and SNR are also held fixed across the splits so that evaluation focuses primarily on generalization to unseen Vietnamese linguistic content.

### Dataset Access

The dataset is available on Hugging Face:

https://huggingface.co/datasets/KhaBui/PESEM-VS

---

## Evaluation Metrics

### PESQ

**Perceptual Evaluation of Speech Quality (PESQ)** measures perceptual speech quality.

Higher is better.

---

### STOI

**Short-Time Objective Intelligibility (STOI)** measures speech intelligibility.

Higher is better.

---

### F0-RMSE

To evaluate tonal preservation, fundamental-frequency contours are extracted from both clean and enhanced speech using **pYIN**.

F0-RMSE is calculated over frames that are identified as voiced in both signals:

\[
F0\text{-RMSE}
=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left(
F_{0}^{clean}(i)
-
F_{0}^{enh}(i)
\right)^2
}
\]

where \(N\) is the number of valid voiced frames.

Lower values indicate closer preservation of the reference pitch contour.

---

### Pitch Failure Rate

F0-RMSE only evaluates frames for which pitch can be successfully estimated.

**Pitch Failure Rate (PFR)** complements F0-RMSE by measuring the percentage of reference voiced frames where the enhanced speech fails to produce a valid corresponding F0 estimate.

Lower is better.

A PFR of **0%** means that a valid pitch estimate is obtained for every reference voiced frame under the evaluation protocol.

---

## Experimental Protocol

The study consists of four experiments.

### Experiment 1 — Zero-Shot Cross-Lingual Evaluation

English-pretrained models are evaluated directly on the Vietnamese test set without Vietnamese adaptation.

This experiment measures cross-lingual generalization in terms of:

- perceptual quality
- intelligibility
- pitch preservation
- pitch-tracking reliability

### Experiment 2 — In-Domain Fine-Tuning

All architectures are fine-tuned on the Vietnamese training partition.

The goal is to determine whether Vietnamese adaptation improves both conventional enhancement metrics and tonal preservation.

### Experiment 3 — Multi-Criteria Model Selection

Models are jointly compared using:

- PESQ
- STOI
- F0-RMSE
- PFR

The comparison is **qualitative rather than based on a weighted aggregate score**.

This avoids allowing strong performance on one metric to compensate for substantial degradation on another dimension, particularly pitch preservation.

### Experiment 4 — Checkpoint Trajectory Analysis

The leading candidates identified during multi-criteria evaluation are analyzed across their fine-tuning checkpoints.

The experiment studies how perceptual quality and pitch preservation evolve during adaptation.

---

# Results

## Zero-Shot Cross-Lingual Evaluation

| Model | PESQ ↑ | STOI ↑ | F0-RMSE Mean (Hz) ↓ | PFR ↓ |
|---|---:|---:|---:|---:|
| FullSubNet | 2.643 | 0.938 | 30.87 | 2.66% |
| FullSubNet+ | 2.749 | 0.947 | 31.92 | 0.64% |
| InterSubNet | 2.863 | 0.949 | 28.09 | 0.74% |
| SEMamba | 1.921 | 0.902 | 67.22 | 1.70% |
| GTCRN | 2.133 | 0.921 | 41.16 | 1.81% |
| MANNER | 1.203 | 0.663 | 128.46 | 5.74% |
| CMGAN | 2.011 | 0.916 | 55.23 | 1.60% |
| **MP-SENet** | **3.283** | **0.960** | **20.89** | 1.38% |
| MetricGAN+ | 1.715 | 0.801 | 202.08 | 5.74% |
| SGMSE | 1.644 | 0.891 | 53.84 | 3.62% |

The zero-shot experiment shows substantial differences among architectures.

MP-SENet obtains the strongest overall zero-shot metric combination among the evaluated models, while several architectures show considerably larger pitch errors despite retaining some speech-enhancement capability.

These results suggest that conventional perceptual performance alone does not fully describe cross-lingual tonal preservation.

---

## Vietnamese In-Domain Fine-Tuning

| Model | PESQ ↑ | STOI ↑ | F0-RMSE Median (Hz) ↓ | F0-RMSE Mean (Hz) ↓ | PFR ↓ |
|---|---:|---:|---:|---:|---:|
| FullSubNet | 2.241 | 0.926 | 4.69 | 42.80 | 0.64% |
| FullSubNet+ | 2.726 | 0.946 | 3.43 | 37.14 | 0.43% |
| InterSubNet | 1.805 | 0.906 | 6.91 | 70.38 | 0.74% |
| SEMamba | 3.354 | 0.964 | 2.99 | 16.90 | 0.32% |
| GTCRN | 2.548 | 0.939 | 3.71 | 30.49 | 0.21% |
| MANNER | 2.212 | 0.921 | 3.91 | 28.75 | 0.11% |
| CMGAN | 3.393 | 0.965 | 2.40 | 15.27 | 0.11% |
| **MP-SENet** | **3.611** | **0.971** | **1.97** | **11.27** | **0.00%** |
| MetricGAN+ | 2.050 | 0.851 | 248.95 | 243.44 | 1.81% |
| SGMSE | 2.943 | 0.955 | 2.45 | 16.82 | **0.00%** |

Fine-tuning produces strongly architecture-dependent behavior.

Among the evaluated architectures:

- MP-SENet achieves PESQ **3.611**, STOI **0.971**, mean F0-RMSE **11.27 Hz**, and **0.00% PFR**.
- CMGAN improves from **55.23 Hz** to **15.27 Hz** mean F0-RMSE.
- SEMamba improves from **67.22 Hz** to **16.90 Hz**.
- SGMSE reaches **16.82 Hz** mean F0-RMSE with **0.00% PFR**.
- MetricGAN+ improves in PESQ but remains strongly pitch-distorted.
- The evaluated sub-band models show limited or negative transfer in pitch preservation after adaptation.

The results therefore indicate that **in-domain fine-tuning alone does not guarantee tonal preservation**.

---

## Multi-Criteria Evaluation

Rather than computing a weighted score, models are qualitatively grouped according to their relative behavior across:

- perceptual quality
- intelligibility
- pitch reconstruction
- pitch failure rate

The analysis identifies **MP-SENet and CMGAN** as the two most consistently balanced candidates across the evaluated dimensions.

These models are therefore selected for checkpoint-level analysis.

---

## Checkpoint Trajectory Analysis

MP-SENet and CMGAN are evaluated across **20 fine-tuning checkpoints**.

CMGAN shows rapid PESQ improvement during the earlier stages of training and approaches a narrow PESQ range around the later checkpoints, while its F0-RMSE continues to decrease.

MP-SENet starts from a stronger zero-shot baseline and maintains lower absolute F0-RMSE than CMGAN throughout the evaluated checkpoints.

Its PESQ continues improving until approximately checkpoint 19, with no comparable plateau observed within the evaluated training horizon.

![Checkpoint trajectory](assets/checkpoint_trajectory.png)

---

# Main Findings

The experiments highlight three main observations.

### 1. Perceptual quality does not guarantee tonal preservation

Models can achieve acceptable PESQ or STOI while introducing substantial F0 distortion.

Pitch-aware evaluation is therefore useful when studying speech enhancement for tonal languages.

### 2. Fine-tuning behavior is architecture-dependent

Vietnamese adaptation improves MP-SENet, CMGAN, SEMamba, and SGMSE across important perceptual and pitch-related dimensions.

In contrast, the evaluated sub-band models show limited or negative pitch-transfer behavior under the same experimental protocol.

### 3. MP-SENet provides the strongest balance in this evaluation

Across the evaluated conditions, MP-SENet provides the strongest overall combination of:

- PESQ
- STOI
- F0-RMSE
- PFR

The checkpoint analysis also shows consistently lower F0-RMSE than CMGAN across the observed training trajectory.

---

# Repository Structure

```text
PESEM-VS/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── data/
│   ├── evaluation/
│   │   ├── metric.py
│   │   └── pfr.py
│   ├── inference/
│   └── utils/
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── evaluate_zero_shot.py
│   ├── evaluate_finetuned.py
│   └── checkpoint_analysis.py
│
├── results/
│   ├── zero_shot.csv
│   ├── finetuned.csv
│   └── checkpoint_trajectories.csv
│
├── assets/
│   ├── pipeline.pdf
│   ├── taxonomy.pdf
│   └── checkpoint_trajectory.pdf
│
├── configs/
│   └── ...
│
└── paper/
    └── PESEM_VS.pdf
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/hkha0801-sketch/PESEM-VS.git
cd PESEM-VS
```

Create a Python virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Reproducing the Experiments

## Prepare Dataset

```bash
python scripts/prepare_dataset.py
```

The dataset preparation pipeline constructs the train/test partitions and prepares the noisy-clean pairs required by the evaluation pipeline.

---

## Zero-Shot Evaluation

```bash
python scripts/evaluate_zero_shot.py
```

Expected result file:

```text
results/zero_shot.csv
```

---

## Fine-Tuned Evaluation

```bash
python scripts/evaluate_finetuned.py
```

Expected result file:

```text
results/finetuned.csv
```

---

## Checkpoint Analysis

```bash
python scripts/checkpoint_analysis.py
```

Expected output:

```text
results/checkpoint_trajectories.csv
```

and the corresponding trajectory visualization:

```text
assets/checkpoint_trajectory.png
```

---

# Visual Overview

## Architecture Taxonomy

![Speech enhancement architecture taxonomy](assets/taxonomy.png)

The evaluated models span sub-band, lightweight/state-space, attention/conformer, and generative/objective-driven approaches.

---

## Experimental Pipeline

![PESEM-VS experimental pipeline](assets/pipeline.png)

The pipeline consists of dataset preparation, zero-shot inference, Vietnamese fine-tuning, objective evaluation, multi-criteria model comparison, and checkpoint analysis.

---

# Limitations

The current evaluation has several limitations.

1. Experiments use a fixed **10 dB SNR**.
2. The corpus contains only **20 core script sentences**.
3. More severe and non-stationary noise conditions remain unexplored.
4. pYIN-based F0-RMSE is evaluated on successfully detected voiced frames and may not fully capture distortions around voiced/unvoiced boundaries.

These limitations should be considered when interpreting the reported results.

---

# Future Work

Future extensions of PESEM-VS will investigate:

- tonal-aware speech enhancement architectures
- phase-aware modeling combined with efficient state-space backbones
- unseen-noise evaluation
- lower-SNR conditions
- standardized speech-enhancement benchmarks
- downstream Vietnamese ASR evaluation

One direction is to combine the phase-aware modeling of **MP-SENet** with efficient state-space architectures such as **SEMamba**.

Enhanced speech will also be evaluated with Vietnamese ASR systems such as **PhoWhisper** to investigate whether improved pitch preservation affects lexical confusion and Word Error Rate.

---

# Paper

The paper associated with this repository is available at:

```text
paper/PESEM_VS.pdf
```

**Title**

> An Empirical Study on Tonal Fidelity Preservation for Speech Enhancement Model Selection in Vietnamese

---

# Citation

If you use PESEM-VS, the dataset, evaluation protocol, or experimental results in your research, please cite the associated paper.

Citation metadata is provided in:

```text
CITATION.cff
```

The final BibTeX entry will be added after publication metadata becomes available.

---

# Dataset

PESEM-VS Dataset:

https://huggingface.co/datasets/KhaBui/PESEM-VS

---

# License

Please see the [`LICENSE`](LICENSE) file for licensing information.

---

# Acknowledgements

This study builds upon the official implementations and pretrained checkpoints of the evaluated speech enhancement architectures.

We thank the authors of FullSubNet, FullSubNet+, InterSubNet, SEMamba, GTCRN, MANNER, CMGAN, MP-SENet, MetricGAN+, and SGMSE for making their research and implementations available to the community.

The noise material used in dataset construction is derived from the **DNS Challenge Noise Dataset**.

---

# Contact

For questions, reproducibility issues, or research discussions, please open a GitHub issue in this repository.

---

<p align="center">
  <b>PESEM-VS</b><br>
  Perceptual Quality × Intelligibility × Tonal Preservation
</p>
