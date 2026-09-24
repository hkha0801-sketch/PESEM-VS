# PESEM-VS

## An Empirical Study on Tonal Fidelity Preservation for Speech Enhancement Model Selection in Vietnamese

**PESEM-VS** is the repository accompanying an empirical study of how modern speech enhancement architectures preserve **Vietnamese tonal information** under zero-shot cross-lingual evaluation and in-domain fine-tuning.

Most contemporary speech enhancement systems are developed and evaluated primarily on English-language corpora. This raises questions about their cross-lingual behavior when applied to tonal languages such as Vietnamese, where fundamental-frequency (**F0**) contours carry lexical information.

A model may improve conventional perceptual metrics while still distorting linguistically meaningful pitch patterns. PESEM-VS therefore evaluates speech enhancement systems jointly in terms of:

- perceptual speech quality,
- speech intelligibility,
- pitch preservation,
- and pitch-tracking reliability.

This repository provides the evaluation code, fine-tuning notebooks, experimental results, figures, configuration files, dataset information, and model checkpoints used in the study.

---

## Research Questions

The study investigates two main questions:

1. **How well do English-pretrained speech enhancement models generalize to Vietnamese speech without adaptation?**

2. **Does in-domain Vietnamese fine-tuning improve both perceptual speech quality and tonal preservation?**

The evaluation uses four complementary metrics:

- **PESQ** — Perceptual Evaluation of Speech Quality
- **STOI** — Short-Time Objective Intelligibility
- **F0-RMSE** — fundamental-frequency reconstruction error
- **PFR** — Pitch Failure Rate

---

## Evaluated Speech Enhancement Architectures

Ten representative architectures are evaluated across four model families.

| Category | Models |
|---|---|
| Sub-band / Multi-stage | FullSubNet, FullSubNet+, InterSubNet |
| Lightweight / Resource-efficient | SEMamba, GTCRN |
| Attention / Conformer-based | MANNER, CMGAN, MP-SENet |
| Generative / Metric-driven | MetricGAN+, SGMSE |

### Architecture Taxonomy

<p align="center">
  <img src="./assets/taxonomy.png" alt="Taxonomy of speech enhancement paradigms" width="760">
</p>

The selected architectures represent different approaches to frequency decomposition, temporal modeling, state-space processing, attention, conformer modeling, perceptual optimization, and generative enhancement.

---

## Dataset

The experimental corpus contains:

- **200 clean Vietnamese recordings**
- **47 environmental noise recordings**
- **9,400 noisy-clean speech pairs**
- **10 dB fixed SNR**
- multiple Vietnamese regional accents
- both synthetic and human speech sources

The 200 clean recordings are constructed from:

```text
20 script sentences
× 2 voice-source types
× 5 regional accents
= 200 clean recordings
```

Each clean recording is mixed with 47 environmental noise recordings:

```text
200 × 47 = 9,400 noisy-clean pairs
```

### Dataset Split

The corpus is split by script identity.

| Split | Scripts | Clean Speech | Noisy Speech |
|---|---:|---:|---:|
| Train | S03–S20 | 180 | 8,460 |
| Test | S01–S02 | 20 | 940 |
| **Total** | **S01–S20** | **200** | **9,400** |

The evaluation set therefore contains unseen sentence content while maintaining comparable accent and voice-source distributions.

Noise inventory and SNR are controlled across the two splits to reduce acoustic confounding.

### Dataset Access

**Hugging Face:**

https://huggingface.co/datasets/KhaBui/PESEM-VS

---

## Experimental Pipeline

<p align="center">
  <img src="./assets/pipeline.png" alt="PESEM-VS experimental pipeline" width="900">
</p>

The study consists of four sequential experiments:

1. **Zero-Shot Cross-Lingual Evaluation**  
   English-pretrained models are evaluated directly on Vietnamese speech without adaptation.

2. **In-Domain Fine-Tuning**  
   All architectures are fine-tuned using the Vietnamese training split.

3. **Multi-Criteria Model Analysis**  
   PESQ, STOI, F0-RMSE, and PFR are jointly considered without using a weighted aggregate score.

4. **Checkpoint Trajectory Analysis**  
   MP-SENet and CMGAN are evaluated across 20 fine-tuning checkpoints.

---

## Evaluation Metrics

### PESQ

**Perceptual Evaluation of Speech Quality (PESQ)** measures perceptual speech quality.

**Higher is better.**

### STOI

**Short-Time Objective Intelligibility (STOI)** measures speech intelligibility.

**Higher is better.**

### F0-RMSE

Fundamental-frequency contours are extracted from clean and enhanced speech using **probabilistic YIN (pYIN)**.

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

where \(N\) is the number of valid voiced frames.

**Lower is better.**

### Pitch Failure Rate (PFR)

PFR complements F0-RMSE by measuring the percentage of reference voiced frames for which the enhanced signal fails to produce a valid corresponding F0 estimate.

```math
\mathrm{PFR}
=
\frac{N_{\mathrm{failure}}}
{N_{\mathrm{reference\ voiced}}}
\times 100\%
```

**Lower is better.**

A PFR of **0%** indicates that a valid F0 estimate is obtained for every reference voiced frame.

---

## Experimental Results

Detailed numerical results are provided directly in the repository.

### Zero-Shot Evaluation

[`results/zero_shot.csv`](./results/zero_shot.csv)

Contains PESQ, STOI, F0-RMSE, and PFR results for all ten architectures before Vietnamese adaptation.

### Fine-Tuned Evaluation

[`results/finetuned.csv`](./results/finetuned.csv)

Contains the corresponding results after Vietnamese in-domain fine-tuning.

### Checkpoint Trajectories

[`results/checkpoint_trajectories.csv`](./results/checkpoint_trajectories.csv)

Contains checkpoint-level results for MP-SENet and CMGAN.

---

## Main Findings

### 1. Perceptual quality does not necessarily imply tonal preservation

Models can obtain favorable PESQ or STOI values while still introducing substantial F0 distortion.

This suggests that conventional perceptual metrics alone may provide an incomplete evaluation of speech enhancement for tonal languages.

---

### 2. Fine-tuning behavior is architecture-dependent

Vietnamese adaptation does not produce uniform improvements across architectures.

Under the evaluated conditions, several attention/conformer, state-space, and generative models improve substantially, whereas the evaluated sub-band models show limited or negative transfer.

---

### 3. MP-SENet shows the most balanced performance under the evaluated conditions

Among the ten evaluated architectures, MP-SENet provides the strongest overall balance across:

- PESQ,
- STOI,
- F0-RMSE,
- and PFR.

After fine-tuning, MP-SENet reaches:

```text
PESQ       : 3.611
STOI       : 0.971
F0-RMSE    : 11.27 Hz
PFR        : 0.00%
```

These results apply specifically to the PESEM-VS experimental conditions and should not be interpreted as a universal ranking across all datasets or operating environments.

---

## Checkpoint Trajectory Analysis

MP-SENet and CMGAN are further evaluated across **20 fine-tuning checkpoints**.

<p align="center">
  <img src="./assets/checkpoint_trajectory.png" alt="Checkpoint trajectories of MP-SENet and CMGAN" width="760">
</p>

CMGAN shows rapid improvement during early fine-tuning and later approaches a relatively narrow PESQ range while its F0-RMSE continues to decrease.

MP-SENet starts from a stronger zero-shot baseline and maintains lower absolute F0-RMSE than CMGAN throughout the evaluated checkpoints.

Its PESQ continues improving until approximately checkpoint 19, with no comparable plateau observed within the evaluated training horizon.

Detailed values are available in:

[`results/checkpoint_trajectories.csv`](./results/checkpoint_trajectories.csv)

---

## Repository Structure

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
│
├── data/
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

Large model-weight files are not stored directly in the Git repository.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/hkha0801-sketch/PESEM-VS.git
cd PESEM-VS
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Reproducing the Experiments

Model-specific training and fine-tuning notebooks are available under:

[`scripts/`](./scripts/)

Common evaluation code is provided under:

[`src/`](./src/)

including:

- [`src/metrics.py`](./src/metrics.py)
- [`src/pfr.py`](./src/pfr.py)

A typical experiment follows:

```text
Vietnamese dataset
        ↓
Pretrained speech enhancement model
        ↓
Zero-shot evaluation
        ↓
Vietnamese fine-tuning
        ↓
Enhanced test speech
        ↓
PESQ / STOI / F0-RMSE / PFR
```

Because the evaluated architectures originate from different implementations, model-specific dependencies and initialization procedures are documented in the corresponding notebooks.

---

## Model Checkpoints

Model checkpoints are stored externally to keep the Git repository lightweight.

**Google Drive:**

[PESEM-VS Checkpoints](https://drive.google.com/drive/folders/1juuIjFd1qkZ4gYRTwWHAXudvqHP4F89f?usp=drive_link)

---

## Reproducibility Notes

The evaluated architectures differ in:

- original implementation,
- dependency versions,
- pretrained initialization,
- checkpoint format,
- preprocessing,
- and training procedure.

PESEM-VS therefore separates common evaluation code from model-specific fine-tuning workflows.

For reproducing a particular architecture, refer to:

1. the corresponding notebook under [`scripts/`](./scripts/),
2. its configuration,
3. the provided checkpoint,
4. and the original implementation of the architecture.

---

## Limitations

The current study is conducted under a controlled experimental setting.

Important limitations include:

- evaluation at a fixed **10 dB SNR**,
- a corpus based on **20 core Vietnamese sentences**,
- reuse of the same noise inventory across training and evaluation,
- and F0 evaluation based on pYIN-derived voiced frames.

The **9,400 noisy-clean pairs** represent multiple acoustic mixtures derived from 200 clean recordings rather than 9,400 independent linguistic utterances.

The results should therefore be interpreted within the experimental conditions used in PESEM-VS.

---

## Future Work

Future work may extend PESEM-VS through:

- lower and varying SNR conditions,
- unseen noise distributions,
- broader Vietnamese linguistic coverage,
- additional speaker and accent diversity,
- tonal-aware enhancement objectives,
- phase-aware and state-space hybrid architectures,
- subjective listening tests,
- and downstream Vietnamese ASR evaluation.

One planned direction is to evaluate enhanced speech using Vietnamese ASR systems such as **PhoWhisper** to examine whether improved F0 preservation reduces lexical confusion and Word Error Rate.

---

## Resources

### Repository

https://github.com/hkha0801-sketch/PESEM-VS

### Dataset

https://huggingface.co/datasets/KhaBui/PESEM-VS

### Model Checkpoints

https://drive.google.com/drive/folders/1juuIjFd1qkZ4gYRTwWHAXudvqHP4F89f?usp=drive_link

### Experimental Results

- [Zero-shot results](./results/zero_shot.csv)
- [Fine-tuned results](./results/finetuned.csv)
- [Checkpoint trajectories](./results/checkpoint_trajectories.csv)

### Evaluation Code

- [Metric implementation](./src/metrics.py)
- [PFR implementation](./src/pfr.py)

### Fine-Tuning Scripts

- [Model-specific notebooks](./scripts/)

---

## Paper

This repository accompanies the manuscript:

> **An Empirical Study on Tonal Fidelity Preservation for Speech Enhancement Model Selection in Vietnamese**

The work evaluates ten representative speech enhancement architectures using a joint **PESQ, STOI, F0-RMSE, and PFR** evaluation protocol.

Publication information will be added when official bibliographic metadata becomes available.

---

## Citation

If you use the PESEM-VS dataset, code, experimental protocol, or results, please cite the corresponding paper.

The official citation will be updated after publication information becomes available.

```bibtex
@inproceedings{pesemvs2026,
  title  = {An Empirical Study on Tonal Fidelity Preservation for Speech Enhancement Model Selection in Vietnamese},
  author = {Bui, Kha and Mai, Khang Trong},
  year   = {2026},
  note   = {Publication information to be updated}
}
```

---

## Acknowledgements

PESEM-VS builds on publicly available speech enhancement architectures and evaluation methods developed by the speech-processing research community.

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

Users reproducing or extending the experiments should also cite the original publications corresponding to the architectures they use.

PESEM-VS focuses on empirical cross-architecture evaluation and does not claim ownership of the original speech enhancement models.
