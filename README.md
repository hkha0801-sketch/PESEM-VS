
##An Empirical Study on Tonal Fidelity Preservation for Speech Enhancement Model Selection in Vietnamese

**PESEM-VS** is the repository accompanying an empirical study of how modern speech enhancement architectures preserve **Vietnamese tonal information** under zero-shot cross-lingual evaluation and in-domain fine-tuning.

Most contemporary speech enhancement systems are developed and evaluated primarily on English-language corpora. This raises questions about their cross-lingual behavior when applied to tonal languages such as Vietnamese, where fundamental-frequency (**F0**) contours carry lexical information.

A model may improve conventional perceptual metrics while still distorting pitch patterns that are linguistically meaningful in Vietnamese.

This study therefore evaluates speech enhancement systems jointly in terms of:

- perceptual speech quality,
- speech intelligibility,
- pitch preservation,
- and pitch-tracking reliability.

The repository provides evaluation code, fine-tuning notebooks, experimental results, figures, configuration files, and supporting material used in the PESEM-VS study.

---

## Research Questions

The study investigates two main questions:

1. **How well do English-pretrained speech enhancement models generalize to Vietnamese speech without adaptation?**

2. **Does in-domain Vietnamese fine-tuning improve both perceptual speech quality and tonal preservation?**

To answer these questions, ten representative speech enhancement architectures are evaluated under a unified protocol using four complementary metrics:

- **PESQ** — Perceptual Evaluation of Speech Quality
- **STOI** — Short-Time Objective Intelligibility
- **F0-RMSE** — fundamental-frequency reconstruction error
- **PFR** — Pitch Failure Rate

---

## Evaluated Speech Enhancement Architectures

The study evaluates ten architectures spanning four representative model families.

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

The selected models represent different design choices in frequency decomposition, temporal modeling, attention, state-space processing, perceptual optimization, and generative enhancement.

---

## Dataset

The experimental corpus contains:

- **200 clean Vietnamese utterances**
- **47 environmental noise recordings**
- **9,400 noisy-clean speech pairs**
- **10 dB fixed SNR**
- multiple Vietnamese regional accents
- both synthetic and human speech sources

The 200 clean recordings are constructed from:

- **20 script sentences**
- **2 voice-source types**
  - synthetic speech
  - human recordings
- **5 regional accents per source**

Each clean utterance is mixed with 47 noise recordings, producing:

```text
200 clean utterances × 47 noises = 9,400 noisy-clean pairs
