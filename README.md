# PESEM-VS

## An Empirical Study on Tonal Fidelity Preservation for Speech Enhancement Model Selection in Vietnamese

<p align="center">
  <b>Cross-architecture Evaluation of Speech Enhancement Models for Vietnamese Tonal Speech</b>
</p>

<p align="center">
  Zero-shot Evaluation • In-domain Fine-tuning • Tonal Preservation • F0-RMSE • Pitch Failure Rate
</p>

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Research Questions](#research-questions)
- [Main Contributions](#main-contributions)
- [Evaluated Speech Enhancement Architectures](#evaluated-speech-enhancement-architectures)
- [Model Selection](#model-selection)
- [Dataset](#dataset)
  - [Dataset Construction](#dataset-construction)
  - [Dataset Split](#dataset-split)
  - [Dataset Access](#dataset-access)
- [Experimental Pipeline](#experimental-pipeline)
- [Experimental Design](#experimental-design)
- [Evaluation Metrics](#evaluation-metrics)
  - [PESQ](#pesq)
  - [STOI](#stoi)
  - [F0-RMSE](#f0-rmse)
  - [Pitch Failure Rate](#pitch-failure-rate-pfr)
- [Experimental Results](#experimental-results)
  - [Zero-Shot Cross-Lingual Evaluation](#1-zero-shot-cross-lingual-evaluation)
  - [In-Domain Fine-Tuning](#2-in-domain-fine-tuning)
  - [Multi-Criteria Analysis](#3-multi-criteria-analysis)
  - [Checkpoint Trajectory Analysis](#4-checkpoint-trajectory-analysis)
- [Main Findings](#main-findings)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Results Files](#results-files)
- [Model Checkpoints](#model-checkpoints)
- [Reproducibility Notes](#reproducibility-notes)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Resources](#resources)
- [References to Evaluated Architectures](#references-to-evaluated-architectures)
- [Paper](#paper)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

# Overview

**PESEM-VS** is the repository accompanying the empirical study:

> **An Empirical Study on Tonal Fidelity Preservation for Speech Enhancement Model Selection in Vietnamese**

The project investigates how modern speech enhancement architectures preserve **Vietnamese tonal information** under two evaluation settings:

1. **Zero-shot cross-lingual evaluation**
2. **In-domain Vietnamese fine-tuning**

Most contemporary speech enhancement systems are developed and evaluated primarily on English-language corpora.

This raises an important question when such models are applied to Vietnamese.

Vietnamese is a tonal language in which fundamental-frequency (**F0**) contours carry lexical information. Therefore, a speech enhancement model may improve conventional perceptual metrics while still modifying pitch patterns that are linguistically meaningful.

PESEM-VS evaluates speech enhancement systems jointly in terms of:

- perceptual speech quality,
- speech intelligibility,
- pitch reconstruction,
- and pitch-tracking reliability.

The repository provides:

- evaluation code,
- common metric implementations,
- model-specific fine-tuning notebooks,
- experimental configurations,
- zero-shot results,
- fine-tuned results,
- checkpoint trajectories,
- figures used in the study,
- dataset information,
- and external model checkpoints.

---

# Motivation

Deep-learning-based speech enhancement has achieved substantial progress in:

- noise suppression,
- perceptual speech quality,
- speech intelligibility,
- and robustness.

However, many widely used architectures are developed primarily using English-language speech.

Applying these models directly to Vietnamese creates a cross-lingual setting in which language-dependent acoustic characteristics may differ from those observed during training.

This is particularly relevant for Vietnamese because **F0 trajectories contribute to lexical tone**.

A model may successfully suppress noise while altering the pitch trajectory of the underlying speech signal.

Consequently:

```text
Good perceptual enhancement
        does not necessarily imply
Good tonal preservation
```

Traditional speech-enhancement evaluation therefore provides only part of the information required for tonal-language speech processing.

PESEM-VS combines conventional perceptual metrics with pitch-aware evaluation to study this issue empirically.

---

# Research Questions

The study investigates two primary research questions.

## RQ1 — Zero-Shot Cross-Lingual Generalization

**How well do English-pretrained speech enhancement models generalize to Vietnamese speech without adaptation?**

The objective is to evaluate whether pretrained models retain useful enhancement capability when directly applied to Vietnamese speech.

The experiment considers both:

- conventional speech-enhancement performance,
- and preservation of Vietnamese pitch information.

---

## RQ2 — In-Domain Vietnamese Adaptation

**Does in-domain Vietnamese fine-tuning improve both perceptual speech quality and tonal preservation?**

All selected architectures are fine-tuned using Vietnamese speech and reevaluated on held-out Vietnamese data.

This allows the study to examine whether adaptation improves:

- PESQ,
- STOI,
- F0-RMSE,
- PFR,

and whether adaptation behavior differs across architectural families.

---

# Main Contributions

The study provides the following empirical contributions.

## 1. Cross-Architecture Evaluation

Ten representative speech enhancement architectures are evaluated under the same Vietnamese experimental protocol.

The selected models cover multiple architectural paradigms:

- sub-band processing,
- lightweight recurrent processing,
- state-space modeling,
- attention,
- conformer-based modeling,
- metric-driven optimization,
- and generative enhancement.

---

## 2. Pitch-Aware Evaluation

Conventional speech-enhancement metrics are evaluated together with:

- **F0-RMSE**
- **Pitch Failure Rate (PFR)**

These measurements provide complementary information about pitch preservation that may not be captured by PESQ and STOI alone.

---

## 3. Zero-Shot and Fine-Tuned Comparison

Each architecture is evaluated:

```text
English-pretrained model
        ↓
Zero-shot Vietnamese evaluation
        ↓
Vietnamese fine-tuning
        ↓
Fine-tuned Vietnamese evaluation
```

This enables direct analysis of architecture-dependent adaptation.

---

## 4. Multi-Criteria Evaluation

The study jointly considers:

- perceptual quality,
- intelligibility,
- F0 accuracy,
- pitch-tracking reliability.

The models are not reduced to a single weighted aggregate score.

---

## 5. Checkpoint-Level Analysis

The strongest balanced candidates are additionally analyzed across **20 fine-tuning checkpoints**.

This provides insight into:

- convergence behavior,
- PESQ evolution,
- F0-RMSE evolution,
- and architecture-specific adaptation trajectories.

---

# Evaluated Speech Enhancement Architectures

The study evaluates ten architectures spanning four representative model families.

| Category | Models |
|---|---|
| **Sub-band / Multi-stage** | FullSubNet, FullSubNet+, InterSubNet |
| **Lightweight / Resource-efficient** | SEMamba, GTCRN |
| **Attention / Conformer-based** | MANNER, CMGAN, MP-SENet |
| **Generative / Metric-driven** | MetricGAN+, SGMSE |

---

## Architecture Taxonomy

<p align="center">
  <img src="./assets/taxonomy.png"
       alt="Taxonomy of speech enhancement paradigms"
       width="760">
</p>

The selected models represent different design choices in:

- frequency decomposition,
- temporal modeling,
- state-space processing,
- recurrent modeling,
- attention,
- conformer processing,
- magnitude/phase reconstruction,
- perceptual optimization,
- and generative denoising.

---

# Model Selection

The final ten architectures were selected from a broader pool of fifteen candidate systems.

Three main criteria were considered.

## Criterion 1 — Task Compatibility

Models should be suitable for:

- single-source speech,
- single-channel input,
- speech denoising / enhancement.

Models primarily designed for fundamentally different tasks were excluded.

---

## Criterion 2 — Sampling-Rate Compatibility

The PESEM-VS experimental pipeline operates at:

```text
16 kHz
```

Models whose standard processing pipeline was incompatible with the common 16 kHz setup were not included in the final comparison.

---

## Criterion 3 — Architectural Relevance

The selected architectures should provide meaningful architectural diversity for investigating F0-preservation behavior.

Architectures providing little additional information relative to already selected models were not prioritized.

---

## Candidate Architecture Summary

| Model | Status | Category / Selection Rationale |
|---|---|---|
| FullSubNet | Included | Sub-band architecture |
| FullSubNet+ | Included | Sub-band architecture |
| InterSubNet | Included | Sub-band interaction |
| SEMamba | Included | Lightweight / state-space architecture |
| GTCRN | Included | Lightweight recurrent architecture |
| MANNER | Included | Attention-based architecture |
| CMGAN | Included | Conformer-based architecture |
| MP-SENet | Included | Magnitude-phase conformer architecture |
| MetricGAN+ | Included | Metric-driven architecture |
| SGMSE | Included | Generative architecture |
| Conv-TasNet | Excluded | Primarily designed for speech separation |
| DeepFilterNet | Excluded | Sampling-rate / pipeline mismatch |
| Spiking-FullSubNet | Excluded | Different training paradigm |
| LiSenNet | Excluded | Limited additional architectural insight |
| FSPEN | Excluded | Limited additional architectural insight |

---

# Dataset

The PESEM-VS experimental corpus contains:

- **200 clean Vietnamese utterances**
- **47 environmental noise recordings**
- **9,400 noisy-clean speech pairs**
- **10 dB fixed SNR**
- multiple Vietnamese regional accents
- both synthetic and human speech sources

---

# Dataset Construction

The clean-speech corpus consists of:

```text
20 script sentences
×
2 voice-source types
×
5 regional accents
=
200 clean recordings
```

The two voice-source types are:

- synthetic Vietnamese speech,
- human-recorded Vietnamese speech.

Each clean recording is combined with 47 environmental noise recordings.

Therefore:

```text
200 clean recordings
×
47 noise recordings
=
9,400 noisy-clean speech pairs
```

---

## Speech Domains

The sentence inventory includes speech from several domains, including:

- customer-service speech,
- weather-related content,
- contact-information expressions,
- and code-switched content.

---

## Voice Sources

### Synthetic Speech

Synthetic Vietnamese speech is generated using the EverAI text-to-speech system.

### Human Speech

Human speech is recorded using mobile devices.

---

## Regional Variation

Multiple Vietnamese regional accents are represented.

The dataset includes speech associated with:

- Northern Vietnamese,
- Southern Vietnamese,
- Hue,
- Binh Dinh,

together with an additional regional source depending on speech-source availability.

The objective is to introduce controlled regional variation rather than to construct a comprehensive Vietnamese accent corpus.

---

# Dataset Split

The corpus is divided by **script identity**.

| Split | Scripts | Clean Speech | Noisy Speech |
|---|---:|---:|---:|
| Train | S03–S20 | 180 | 8,460 |
| Test | S01–S02 | 20 | 940 |
| **Total** | **S01–S20** | **200** | **9,400** |

The evaluation split therefore contains linguistic content not present among the training scripts.

The training and evaluation subsets retain the same general:

- accent distribution,
- voice-source distribution.

Noise inventory and SNR are controlled across the two subsets to reduce acoustic confounding.

---

# Dataset Access

The dataset is publicly available on Hugging Face:

**PESEM-VS Dataset**

https://huggingface.co/datasets/KhaBui/PESEM-VS

---

# Experimental Pipeline

<p align="center">
  <img src="./assets/pipeline.png"
       alt="PESEM-VS experimental pipeline"
       width="900">
</p>

The overall experimental workflow can be summarized as:

```text
Vietnamese Clean Speech
        │
        │ + Environmental Noise
        ▼
Vietnamese Noisy Speech
        │
        ├─────────────────────────────┐
        │                             │
        ▼                             ▼
Zero-shot Evaluation        Vietnamese Fine-tuning
        │                             │
        │                             ▼
        │                    Fine-tuned Models
        │                             │
        └──────────────┬──────────────┘
                       ▼
                 Enhanced Speech
                       │
                       ▼
              Objective Evaluation
                       │
        ┌──────────────┼───────────────┐
        │              │               │
      PESQ           STOI          F0-RMSE / PFR
```

---

# Experimental Design

The study consists of four sequential experiments.

---

## Experiment 1 — Zero-Shot Cross-Lingual Evaluation

English-pretrained architectures are evaluated directly on the Vietnamese test set.

No Vietnamese adaptation is performed.

The objective is to quantify cross-lingual behavior in terms of:

- perceptual quality,
- intelligibility,
- pitch reconstruction,
- pitch-tracking reliability.

---

## Experiment 2 — In-Domain Fine-Tuning

All ten architectures are fine-tuned using the Vietnamese training subset.

The adapted models are then evaluated using the same test protocol.

This experiment investigates whether Vietnamese fine-tuning mitigates the limitations observed during zero-shot evaluation.

---

## Experiment 3 — Multi-Criteria Model Analysis

The following metrics are jointly considered:

```text
PESQ
STOI
F0-RMSE
PFR
```

No weighted aggregate score is used.

This prevents strong performance on one metric from automatically compensating for weak performance in another dimension.

---

## Experiment 4 — Checkpoint Trajectory Analysis

The strongest balanced candidates are evaluated across:

```text
20 fine-tuning checkpoints
```

The checkpoint analysis focuses on:

- perceptual convergence,
- F0-error convergence,
- adaptation speed,
- and differences between architectures.

MP-SENet and CMGAN are selected for this experiment.

---

# Evaluation Metrics

Four objective metrics are used.

---

## PESQ

**Perceptual Evaluation of Speech Quality (PESQ)** evaluates perceptual speech quality.

```text
Higher = Better
```

---

## STOI

**Short-Time Objective Intelligibility (STOI)** evaluates speech intelligibility.

```text
Higher = Better
```

---

## F0-RMSE

Vietnamese lexical tone is strongly associated with F0 trajectories.

To evaluate pitch preservation, F0 contours are extracted from:

- clean reference speech,
- enhanced speech,

using **probabilistic YIN (pYIN)**.

F0-RMSE is calculated over frames identified as voiced in both signals:

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

where:

```text
N = number of valid voiced frames
```

Interpretation:

```text
Lower F0-RMSE = closer preservation of the reference F0 contour
```

---

## Pitch Failure Rate (PFR)

F0-RMSE only evaluates frames for which valid F0 estimates are available.

**Pitch Failure Rate (PFR)** complements F0-RMSE by measuring the percentage of reference voiced frames for which enhanced speech fails to provide a valid corresponding F0 estimate.

```math
\mathrm{PFR}
=
\frac{
N_{\mathrm{failure}}
}{
N_{\mathrm{reference\ voiced}}
}
\times 100\%
```

Interpretation:

```text
Lower PFR = Better
```

A PFR of:

```text
0%
```

indicates that a valid corresponding F0 estimate is obtained for every reference voiced frame.

Therefore:

```text
F0-RMSE
→ measures pitch reconstruction error when pitch is available.

PFR
→ measures the frequency of pitch-tracking failure.
```

---

# Experimental Results

# 1. Zero-Shot Cross-Lingual Evaluation

The complete numerical results are available at:

[`results/zero_shot.csv`](./results/zero_shot.csv)

---

## Zero-Shot Results

| Model | PESQ | STOI | F0-RMSE Mean (Hz) | PFR |
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

---

## Zero-Shot Observations

Among the evaluated systems, MP-SENet provides the strongest zero-shot combination of:

- PESQ,
- STOI,
- and F0-RMSE.

Its results are:

```text
PESQ       : 3.283
STOI       : 0.960
F0-RMSE    : 20.89 Hz
PFR        : 1.38%
```

The zero-shot evaluation also reveals that acceptable conventional enhancement performance does not necessarily imply reliable F0 preservation.

---

# 2. In-Domain Fine-Tuning

The complete fine-tuning results are available at:

[`results/finetuned.csv`](./results/finetuned.csv)

---

## Fine-Tuned Results

| Model | PESQ | STOI | F0-RMSE Median (Hz) | F0-RMSE Mean (Hz) | PFR |
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
| SGMSE | 2.943 | 0.955 | 2.45 | 16.82 | 0.00% |

---

## MP-SENet

After Vietnamese fine-tuning:

```text
PESQ             : 3.611
STOI             : 0.971
F0-RMSE Median   : 1.97 Hz
F0-RMSE Mean     : 11.27 Hz
PFR              : 0.00%
```

Within the PESEM-VS experimental setting, MP-SENet provides the most balanced performance across the evaluated perceptual and pitch-related metrics.

---

## CMGAN

CMGAN shows substantial adaptation after Vietnamese fine-tuning.

### Zero-shot

```text
PESQ       : 2.011
F0-RMSE    : 55.23 Hz
```

### Fine-tuned

```text
PESQ       : 3.393
STOI       : 0.965
F0-RMSE    : 15.27 Hz
PFR        : 0.11%
```

CMGAN therefore shows a substantial reduction in pitch error after Vietnamese adaptation.

---

## SEMamba

SEMamba also improves substantially after adaptation.

### Zero-shot

```text
F0-RMSE    : 67.22 Hz
```

### Fine-tuned

```text
PESQ       : 3.354
STOI       : 0.964
F0-RMSE    : 16.90 Hz
PFR        : 0.32%
```

Its performance indicates that state-space-based architectures can remain competitive under the PESEM-VS setting.

---

## SGMSE

SGMSE reaches:

```text
PESQ       : 2.943
STOI       : 0.955
F0-RMSE    : 16.82 Hz
PFR        : 0.00%
```

after Vietnamese fine-tuning.

---

## MetricGAN+

MetricGAN+ illustrates an important mismatch between perceptual optimization and tonal fidelity.

### Zero-shot

```text
PESQ       : 1.715
F0-RMSE    : 202.08 Hz
```

### Fine-tuned

```text
PESQ       : 2.050
STOI       : 0.851
F0-RMSE    : 243.44 Hz
PFR        : 1.81%
```

Although PESQ improves after adaptation, pitch reconstruction remains substantially degraded.

---

## Sub-band Models

The evaluated sub-band architectures show limited or negative transfer under the current protocol.

### FullSubNet

```text
Zero-shot F0-RMSE   : 30.87 Hz
Fine-tuned F0-RMSE  : 42.80 Hz
```

### FullSubNet+

```text
Zero-shot F0-RMSE   : 31.92 Hz
Fine-tuned F0-RMSE  : 37.14 Hz
```

### InterSubNet

```text
Zero-shot F0-RMSE   : 28.09 Hz
Fine-tuned F0-RMSE  : 70.38 Hz
```

These results indicate that in-domain fine-tuning does not produce uniform improvements across architectural families.

---

# 3. Multi-Criteria Analysis

The analysis jointly considers:

| Dimension | Metric |
|---|---|
| Perceptual quality | PESQ |
| Intelligibility | STOI |
| Pitch reconstruction | F0-RMSE |
| Pitch-tracking reliability | PFR |

A weighted aggregate score is intentionally not used.

This prevents:

```text
Strong performance on one metric
```

from automatically compensating for:

```text
Poor performance on another metric
```

particularly poor tonal preservation.

Under the evaluated conditions, MP-SENet and CMGAN provide the most consistent balance across the four metrics and are therefore selected for checkpoint-level analysis.

This should be interpreted as an empirical comparison under the PESEM-VS protocol rather than a universal ranking of speech enhancement architectures.

---

# 4. Checkpoint Trajectory Analysis

MP-SENet and CMGAN are analyzed across:

```text
20 fine-tuning checkpoints
```

The numerical results are available at:

[`results/checkpoint_trajectories.csv`](./results/checkpoint_trajectories.csv)

---

## Checkpoint Trajectory Figure

<p align="center">
  <img src="./assets/checkpoint_trajectory.png"
       alt="Checkpoint trajectories of MP-SENet and CMGAN"
       width="760">
</p>

---

## CMGAN Trajectory

CMGAN begins from:

```text
PESQ       : 2.011
F0-RMSE    : 55.23 Hz
```

PESQ improves rapidly during the earlier fine-tuning stages.

Around checkpoint 14, PESQ approaches a relatively narrow range of approximately:

```text
3.39 – 3.42
```

while F0-RMSE continues to decrease.

At checkpoint 20:

```text
F0-RMSE    : 15.27 Hz
```

representing approximately:

```text
72.4% reduction
```

relative to its zero-shot F0-RMSE.

---

## MP-SENet Trajectory

MP-SENet begins from the stronger zero-shot baseline:

```text
PESQ       : 3.283
F0-RMSE    : 20.89 Hz
```

PESQ continues improving through approximately checkpoint 19.

The observed trajectory reaches:

```text
PESQ       : 3.611
F0-RMSE    : 11.27 Hz
```

Relative to the zero-shot condition:

```text
PESQ improvement       ≈ 10.0%
F0-RMSE reduction      ≈ 46.1%
```

Across the observed checkpoints, MP-SENet maintains lower absolute F0-RMSE than CMGAN.

No comparable PESQ plateau is observed within the evaluated training horizon.

---

# Main Findings

## 1. Perceptual Quality Does Not Necessarily Imply Tonal Preservation

Speech enhancement models may obtain favorable PESQ or STOI values while introducing substantial F0 distortion.

For tonal-language speech enhancement:

```text
PESQ + STOI
```

therefore provide only part of the evaluation picture.

Pitch-aware measurements such as:

```text
F0-RMSE + PFR
```

provide complementary information about preservation of linguistically relevant pitch structure.

---

## 2. Fine-Tuning Behavior Is Architecture-Dependent

Vietnamese in-domain adaptation affects architectures differently.

Under the evaluated conditions, substantial improvements are observed for several architectures, including:

- MP-SENet,
- CMGAN,
- SEMamba,
- SGMSE.

Meanwhile, the evaluated sub-band models show limited or negative transfer in pitch-related performance.

Therefore:

> **In-domain fine-tuning alone does not guarantee tonal preservation.**

---

## 3. MP-SENet Provides the Most Balanced Performance Under the Evaluated Conditions

Among the ten evaluated architectures, MP-SENet reaches:

```text
PESQ       : 3.611
STOI       : 0.971
F0-RMSE    : 11.27 Hz
PFR        : 0.00%
```

It provides the most balanced combination of perceptual quality, intelligibility, F0 preservation, and pitch-tracking reliability within the current PESEM-VS protocol.

This result should not be interpreted as a universal ranking across all languages, datasets, noise environments, or speech-enhancement tasks.

---

## 4. Perceptual Optimization May Not Preserve Pitch

MetricGAN+ illustrates that perceptually oriented optimization does not necessarily preserve tonal information.

Despite improved perceptual performance after adaptation, its F0-RMSE remains substantially larger than those of several other evaluated systems.

This observation motivates joint perceptual and pitch-aware evaluation.

---

# Repository Structure

```text
PESEM-VS/
│
├── README.md
├── requirements.txt
├── dataset.yaml
├── .gitignore
│
├── assets/
│   ├── taxonomy.png
│   ├── pipeline.png
│   └── checkpoint_trajectory.png
│
├── configs/
│   └── [experiment / model configuration files]
│
├── data/
│   └── [local dataset-related files]
│
├── results/
│   ├── zero_shot.csv
│   ├── finetuned.csv
│   └── checkpoint_trajectories.csv
│
├── scripts/
│   └── [model-specific training / fine-tuning notebooks]
│
└── src/
    ├── metrics.py
    └── pfr.py
```

Large model-weight files are intentionally not stored directly in the Git repository.

---

# Installation

Clone PESEM-VS:

```bash
git clone https://github.com/hkha0801-sketch/PESEM-VS.git
```

Enter the repository:

```bash
cd PESEM-VS
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Reproducing the Experiments

Model-specific fine-tuning procedures are available under:

[`scripts/`](./scripts/)

Because the evaluated architectures originate from different codebases, each notebook contains the workflow required for the corresponding model.

A typical experiment follows:

```text
1. Prepare dataset
2. Load pretrained model
3. Configure Vietnamese training split
4. Fine-tune architecture
5. Enhance held-out speech
6. Compute PESQ
7. Compute STOI
8. Compute F0-RMSE
9. Compute PFR
10. Save experimental results
```

Common metric implementations are provided under:

[`src/`](./src/)

including:

- [`src/metrics.py`](./src/metrics.py)
- [`src/pfr.py`](./src/pfr.py)

---

# Results Files

## Zero-Shot Results

[`results/zero_shot.csv`](./results/zero_shot.csv)

Contains zero-shot evaluation results for the ten architectures.

---

## Fine-Tuned Results

[`results/finetuned.csv`](./results/finetuned.csv)

Contains evaluation results after Vietnamese in-domain fine-tuning.

---

## Checkpoint Results

[`results/checkpoint_trajectories.csv`](./results/checkpoint_trajectories.csv)

Contains checkpoint-level measurements for:

- MP-SENet
- CMGAN

across the evaluated fine-tuning trajectory.

---

# Model Checkpoints

Model checkpoints are hosted externally to keep the Git repository lightweight.

### PESEM-VS Checkpoints — Google Drive

https://drive.google.com/drive/folders/1juuIjFd1qkZ4gYRTwWHAXudvqHP4F89f?usp=drive_link

No large checkpoint files are committed directly to this repository.

---

# Reproducibility Notes

The evaluated speech-enhancement architectures originate from different implementations and therefore differ in:

- dependencies,
- checkpoint formats,
- model initialization,
- preprocessing,
- training procedures,
- and configuration requirements.

PESEM-VS therefore separates:

```text
Common Evaluation
```

from:

```text
Model-specific Training / Fine-tuning
```

Shared evaluation code is located in:

[`src/`](./src/)

Model-specific workflows are located in:

[`scripts/`](./scripts/)

Users attempting to reproduce a specific architecture should consult:

1. the corresponding PESEM-VS notebook,
2. the model configuration,
3. the original model implementation,
4. and its pretrained checkpoint requirements.

---

# Limitations

The current study is conducted under a controlled experimental protocol.

Several limitations should therefore be considered.

## 1. Fixed SNR

Experiments are performed at:

```text
10 dB SNR
```

The current study does not evaluate more severe or highly variable SNR conditions.

---

## 2. Limited Linguistic Coverage

The corpus is based on:

```text
20 core Vietnamese sentences
```

Although these sentences are realized across multiple voice sources, accents, and noise combinations, the number of independent linguistic scripts remains limited.

---

## 3. Number of Mixtures vs. Independent Utterances

The:

```text
9,400 noisy-clean pairs
```

are generated from:

```text
200 clean recordings × 47 noise recordings
```

Therefore, the 9,400 pairs represent multiple acoustic mixtures rather than 9,400 independent linguistic utterances.

---

## 4. Controlled Noise Conditions

Noise inventory and SNR are intentionally controlled between the training and evaluation splits.

This helps reduce acoustic confounding when examining unseen Vietnamese sentence content.

However, the current experiment does not establish generalization to completely unseen noise distributions.

---

## 5. Pitch Extraction

Pitch analysis is based on pYIN.

F0-RMSE evaluates frames for which valid pitch estimates exist in both clean and enhanced speech.

PFR captures additional pitch-estimation failures.

Nevertheless, this evaluation may not fully capture distortions around:

- voiced/unvoiced boundaries,
- unstable phonation,
- or other complex F0 behavior.

---

## 6. Objective Evaluation

The study focuses primarily on objective evaluation.

Large-scale subjective listening tests are not included.

Therefore, improved F0 measurements should not automatically be interpreted as direct evidence of improved human lexical perception.

---

# Future Work

Several directions can extend the PESEM-VS study.

## Variable and Lower SNR Conditions

Future experiments can study multiple SNR conditions, including more challenging environments.

For example:

```text
0 dB
5 dB
10 dB
15 dB
```

---

## Unseen Noise Evaluation

Future evaluation can introduce noise samples that are completely unseen during Vietnamese fine-tuning.

---

## Larger Vietnamese Corpus

Future datasets can increase:

- linguistic diversity,
- number of sentences,
- speaker diversity,
- accent diversity,
- recording environments.

---

## Tonal-Aware Speech Enhancement

Potential directions include:

- F0-guided input features,
- pitch-aware auxiliary objectives,
- tonal-preservation losses,
- explicit phase modeling,
- multi-task learning.

---

## Phase-Aware + State-Space Modeling

An additional research direction is combining:

- phase-aware modeling similar to **MP-SENet**
- with efficient state-space modeling such as **SEMamba**.

---

## Vietnamese ASR Evaluation

Enhanced speech can be evaluated through downstream Vietnamese ASR systems such as PhoWhisper.

This can investigate whether improved pitch preservation contributes to:

- fewer lexical confusions,
- improved recognition,
- and lower Word Error Rate (WER).

---

## Human Evaluation

Future subjective experiments can evaluate:

- naturalness,
- intelligibility,
- tonal correctness,
- and lexical perception.

---

# Resources

## PESEM-VS Repository

https://github.com/hkha0801-sketch/PESEM-VS

## PESEM-VS Dataset

https://huggingface.co/datasets/KhaBui/PESEM-VS

## PESEM-VS Checkpoints

https://drive.google.com/drive/folders/1juuIjFd1qkZ4gYRTwWHAXudvqHP4F89f?usp=drive_link

---

# References to Evaluated Architectures

The following publications provide the original descriptions of the evaluated architectures.

### FullSubNet

**FullSubNet: A Full-band and Sub-band Fusion Model for Real-time Single-channel Speech Enhancement**

https://arxiv.org/abs/2010.15508

---

### FullSubNet+

**FullSubNet+: Channel Attention FullSubNet with Complex Spectrograms for Speech Enhancement**

https://arxiv.org/abs/2203.12188

---

### InterSubNet

**Inter-SubNet: Speech Enhancement with Subband Interaction**

https://arxiv.org/abs/2305.05599

---

### SEMamba

**An Investigation of Incorporating Mamba for Speech Enhancement**

https://arxiv.org/abs/2405.06573

---

### GTCRN

**GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources**

Publication: ICASSP 2024.

---

### MANNER

**MANNER: Multi-view Attention Network for Noise Erasure**

https://arxiv.org/abs/2203.02181

---

### CMGAN

**CMGAN: Conformer-based Metric GAN for Speech Enhancement**

https://arxiv.org/abs/2203.15149

Extended version:

https://arxiv.org/abs/2209.11112

---

### MP-SENet

**MP-SENet: A Speech Enhancement Model with Parallel Denoising of Magnitude and Phase Spectra**

https://arxiv.org/abs/2305.13686

---

### MetricGAN+

**MetricGAN+: An Improved Version of MetricGAN for Speech Enhancement**

https://arxiv.org/abs/2104.03538

---

### SGMSE

**Speech Enhancement and Dereverberation with Diffusion-based Generative Models**

IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023.

---

# Related Evaluation Methods

## PESQ

A. W. Rix, J. G. Beerends, M. P. Hollier, and A. P. Hekstra.

**Perceptual Evaluation of Speech Quality (PESQ) — A New Method for Speech Quality Assessment of Telephone Networks and Codecs**

ICASSP, 2001.

---

## STOI

C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen.

**A Short-Time Objective Intelligibility Measure for Time-Frequency Weighted Noisy Speech**

ICASSP, 2010.

---

## pYIN

M. Mauch and S. Dixon.

**pYIN: A Fundamental Frequency Estimator Using Probabilistic Threshold Distributions**

ICASSP, 2014.

---

## DNS Challenge Dataset

C. K. A. Reddy et al.

**The INTERSPEECH 2020 Deep Noise Suppression Challenge: Datasets, Subjective Speech Quality and Testing Framework**

https://arxiv.org/abs/2005.13981

---

# Paper

This repository accompanies:

> **An Empirical Study on Tonal Fidelity Preservation for Speech Enhancement Model Selection in Vietnamese**

The study evaluates ten representative speech-enhancement architectures using the joint:

```text
PESQ
STOI
F0-RMSE
PFR
```

evaluation protocol.

The experiments study:

- zero-shot cross-lingual generalization,
- Vietnamese in-domain fine-tuning,
- architecture-dependent adaptation,
- multi-criteria evaluation,
- and checkpoint-level convergence.

Publication information will be added when official bibliographic metadata becomes available.

---

# Citation

If you use the PESEM-VS dataset, experimental protocol, evaluation implementation, or reported results, please cite the corresponding paper.

The official citation will be updated after publication information becomes available.

```bibtex
@inproceedings{pesemvs2026,
  title     = {An Empirical Study on Tonal Fidelity Preservation for Speech Enhancement Model Selection in Vietnamese},
  author    = {Bui, Kha and Mai, Khang Trong},
  year      = {2026},
  note      = {Publication information to be updated}
}
```

> **Note:** The BibTeX entry above is temporary and should be replaced with the official publication metadata once available.

---

# Acknowledgements

PESEM-VS builds on publicly available speech-enhancement architectures and evaluation methods developed by the speech-processing research community.

The evaluated architectures include:

- FullSubNet
- FullSubNet+
- InterSubNet
- SEMamba
- GTCRN
- MANNER
- CMGAN
- MP-SENet
- MetricGAN+
- SGMSE

Users reproducing or extending PESEM-VS experiments should also cite the original publication corresponding to each architecture.

PESEM-VS focuses on empirical cross-architecture evaluation and does not claim ownership of the original speech-enhancement architectures.

---

# Summary

PESEM-VS investigates whether modern speech-enhancement systems preserve tonal information when applied to Vietnamese speech.

The central experimental observation is:

> **Improving conventional perceptual speech-enhancement metrics does not necessarily guarantee preservation of linguistically relevant pitch information.**

Under the evaluated PESEM-VS conditions:

- zero-shot behavior varies substantially across architectures,
- Vietnamese fine-tuning is architecture-dependent,
- several models improve both perceptual and pitch-related performance,
- some architectures exhibit limited or negative pitch-related transfer,
- and MP-SENet provides the most balanced performance among the evaluated models.

The study therefore motivates evaluating Vietnamese and other tonal-language speech-enhancement systems using:

```text
Perceptual evaluation
        +
Intelligibility evaluation
        +
Pitch-aware evaluation
```
