"""
Pitch Failure Rate (PFR) for PESEM-VS.

Definition used in the paper:
PFR is the percentage of reference voiced frames for which the enhanced
signal fails to produce a valid corresponding F0 estimate.

Lower is better. A PFR of 0% means that a valid F0 estimate is obtained
for every reference voiced frame.

This implementation uses librosa.pyin for F0 extraction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np


def extract_f0_pyin(
    audio: np.ndarray,
    sr: int,
    *,
    fmin: float = 50.0,
    fmax: float = 500.0,
    frame_length: int = 2048,
    hop_length: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract F0 and voiced/unvoiced decisions with pYIN.

    Parameters
    ----------
    audio:
        Mono waveform.
    sr:
        Sampling rate.
    fmin, fmax:
        F0 search range in Hz.
    frame_length:
        Analysis frame length.
    hop_length:
        Hop size between frames.

    Returns
    -------
    f0:
        Estimated F0 in Hz. Unvoiced/invalid frames are NaN.
    voiced_flag:
        Boolean array indicating frames classified as voiced by pYIN.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1:
        raise ValueError("audio must be a mono 1-D waveform")

    if audio.size == 0:
        raise ValueError("audio is empty")

    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    if f0 is None or voiced_flag is None:
        raise RuntimeError("pYIN failed to return F0 estimates")

    return np.asarray(f0), np.asarray(voiced_flag, dtype=bool)


def pitch_failure_rate(
    clean_f0: np.ndarray,
    clean_voiced: np.ndarray,
    enhanced_f0: np.ndarray,
    enhanced_voiced: np.ndarray | None = None,
) -> float:
    """
    Compute Pitch Failure Rate (PFR) in percent.

    PFR = 100 * (# reference voiced frames with invalid enhanced F0)
                / (# reference voiced frames)

    A frame is considered a pitch failure when:
      1. the clean/reference frame is voiced, and
      2. the enhanced F0 is NaN/non-finite, or enhanced_voiced is False.

    Parameters
    ----------
    clean_f0:
        Reference F0 contour.
    clean_voiced:
        Reference voiced/unvoiced mask.
    enhanced_f0:
        Enhanced-speech F0 contour.
    enhanced_voiced:
        Optional enhanced voiced/unvoiced mask.

    Returns
    -------
    float
        PFR in percent.

    Notes
    -----
    If there are no voiced reference frames, NaN is returned because
    PFR is undefined for that utterance.
    """
    clean_f0 = np.asarray(clean_f0)
    clean_voiced = np.asarray(clean_voiced, dtype=bool)
    enhanced_f0 = np.asarray(enhanced_f0)

    n = min(len(clean_f0), len(clean_voiced), len(enhanced_f0))

    if enhanced_voiced is not None:
        enhanced_voiced = np.asarray(enhanced_voiced, dtype=bool)
        n = min(n, len(enhanced_voiced))

    clean_f0 = clean_f0[:n]
    clean_voiced = clean_voiced[:n]
    enhanced_f0 = enhanced_f0[:n]

    reference_voiced = clean_voiced & np.isfinite(clean_f0)
    num_reference_voiced = int(reference_voiced.sum())

    if num_reference_voiced == 0:
        return float("nan")

    enhanced_valid = np.isfinite(enhanced_f0)

    if enhanced_voiced is not None:
        enhanced_voiced = enhanced_voiced[:n]
        enhanced_valid &= enhanced_voiced

    failures = reference_voiced & ~enhanced_valid

    return float(100.0 * failures.sum() / num_reference_voiced)


def compute_pfr(
    clean_audio: np.ndarray,
    enhanced_audio: np.ndarray,
    sr: int,
    *,
    fmin: float = 50.0,
    fmax: float = 500.0,
    frame_length: int = 2048,
    hop_length: int = 256,
) -> float:
    """
    Compute PFR directly from clean and enhanced waveforms.
    """
    clean_f0, clean_voiced = extract_f0_pyin(
        clean_audio,
        sr,
        fmin=fmin,
        fmax=fmax,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    enhanced_f0, enhanced_voiced = extract_f0_pyin(
        enhanced_audio,
        sr,
        fmin=fmin,
        fmax=fmax,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    return pitch_failure_rate(
        clean_f0=clean_f0,
        clean_voiced=clean_voiced,
        enhanced_f0=enhanced_f0,
        enhanced_voiced=enhanced_voiced,
    )


def compute_pfr_from_files(
    clean_path: str | Path,
    enhanced_path: str | Path,
    *,
    sr: int = 16000,
    fmin: float = 50.0,
    fmax: float = 500.0,
    frame_length: int = 2048,
    hop_length: int = 256,
) -> float:
    """
    Load two audio files and compute PFR.

    Both files are resampled to the same target sampling rate.
    """
    clean_audio, _ = librosa.load(clean_path, sr=sr, mono=True)
    enhanced_audio, _ = librosa.load(enhanced_path, sr=sr, mono=True)

    # Keep waveform durations aligned before frame-level comparison.
    n_samples = min(len(clean_audio), len(enhanced_audio))
    clean_audio = clean_audio[:n_samples]
    enhanced_audio = enhanced_audio[:n_samples]

    return compute_pfr(
        clean_audio,
        enhanced_audio,
        sr,
        fmin=fmin,
        fmax=fmax,
        frame_length=frame_length,
        hop_length=hop_length,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Pitch Failure Rate (PFR) using pYIN."
    )
    parser.add_argument("--clean", required=True, help="Path to clean/reference WAV")
    parser.add_argument(
        "--enhanced", required=True, help="Path to enhanced/output WAV"
    )
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--fmin", type=float, default=50.0)
    parser.add_argument("--fmax", type=float, default=500.0)
    parser.add_argument("--frame-length", type=int, default=2048)
    parser.add_argument("--hop-length", type=int, default=256)

    args = parser.parse_args()

    pfr = compute_pfr_from_files(
        args.clean,
        args.enhanced,
        sr=args.sr,
        fmin=args.fmin,
        fmax=args.fmax,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
    )

    if np.isnan(pfr):
        print("PFR: NaN (no voiced reference frames)")
    else:
        print(f"PFR: {pfr:.4f}%")


if __name__ == "__main__":
    main()
