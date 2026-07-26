import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import (
    load_config,
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
)

from dataset import SpeechEnhancementDataset, collate_fn
from model import build_model
from losses import get_loss_fn


def run_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    grad_clip,
):
    """
    Train model for one epoch.
    """

    model.train()

    total_loss = 0.0
    n_batches = 0

    for noisy, clean in tqdm(
        loader,
        desc="train",
        leave=False
    ):
        noisy = noisy.to(device)
        clean = clean.to(device)

        # Forward
        est = model(noisy)

        # Đảm bảo output và target có cùng chiều dài
        min_len = min(
            est.shape[-1],
            clean.shape[-1]
        )

        est = est[..., :min_len]
        clean = clean[..., :min_len]

        # Calculate loss
        loss = loss_fn(
            est,
            clean
        )

        # Backward
        optimizer.zero_grad()

        loss.backward()

        # Gradient clipping
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip
            )

        # Update weights
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(
        n_batches,
        1
    )


def main():

    # =========================================================
    # ARGUMENTS
    # =========================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training"
    )

    args = parser.parse_args()


    # =========================================================
    # LOAD CONFIG
    # =========================================================

    cfg = load_config(
        args.config
    )


    # =========================================================
    # SET SEED
    # =========================================================

    set_seed(
        cfg["train"].get(
            "seed",
            42
        )
    )


    # =========================================================
    # DEVICE
    # =========================================================

    device = get_device(
        cfg["train"].get(
            "device",
            "auto"
        )
    )

    print(
        f"[INFO] device = {device}"
    )


    # =========================================================
    # DATASET
    # =========================================================

    data_cfg = cfg["data"]

    print(
        "[INFO] Loading training dataset..."
    )

    train_ds = SpeechEnhancementDataset(
        data_cfg["train_csv"],
        sample_rate=data_cfg["sample_rate"],
        segment_seconds=data_cfg["segment_seconds"],
        train=True,
    )

    print(
        f"[INFO] Number of training samples: {len(train_ds)}"
    )


    # =========================================================
    # DATALOADER
    # =========================================================

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
        drop_last=True,
    )


    # =========================================================
    # MODEL
    # =========================================================

    print(
        "[INFO] Building model..."
    )

    model = build_model(
        cfg
    ).to(device)


    # =========================================================
    # OPTIMIZER
    # =========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get(
            "weight_decay",
            0.0
        ),
    )


    # =========================================================
    # LOSS
    # =========================================================

    loss_fn = get_loss_fn(
        cfg["train"]["loss"],
        stft_cfg=cfg["stft"]
    )


    # =========================================================
    # RESUME TRAINING
    # =========================================================

    start_epoch = 0

    best_train_loss = float(
        "inf"
    )

    if args.resume:

        start_epoch, best_train_loss = load_checkpoint(
            args.resume,
            model,
            optimizer,
            map_location=device
        )

        print(
            f"[INFO] Resumed from: {args.resume}"
        )

        print(
            f"[INFO] Starting epoch: {start_epoch}"
        )

        print(
            f"[INFO] Best train loss: {best_train_loss:.4f}"
        )


    # =========================================================
    # CHECKPOINT DIRECTORY
    # =========================================================

    ckpt_dir = cfg["train"][
        "checkpoint_dir"
    ]

    os.makedirs(
        ckpt_dir,
        exist_ok=True
    )


    # =========================================================
    # TENSORBOARD
    # =========================================================

    writer = SummaryWriter(
        cfg["train"].get(
            "log_dir",
            "runs"
        )
    )


    # =========================================================
    # TRAINING
    # =========================================================

    epochs = cfg["train"][
        "epochs"
    ]

    grad_clip = cfg["train"].get(
        "grad_clip"
    )


    print(
        "\n========================================"
    )

    print(
        "START TRAINING"
    )

    print(
        "========================================"
    )

    print(
        f"Device       : {device}"
    )

    print(
        f"Epochs       : {epochs}"
    )

    print(
        f"Batch size   : {cfg['train']['batch_size']}"
    )

    print(
        f"Learning rate: {cfg['train']['lr']}"
    )

    print(
        f"Loss         : {cfg['train']['loss']}"
    )

    print(
        "========================================\n"
    )


    for epoch in range(
        start_epoch,
        epochs
    ):

        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            grad_clip=grad_clip,
        )


        # =====================================================
        # PRINT LOG
        # =====================================================

        print(
            f"[Epoch {epoch + 1}/{epochs}] "
            f"train_loss={train_loss:.4f}"
        )


        # =====================================================
        # TENSORBOARD
        # =====================================================

        writer.add_scalar(
            "loss/train",
            train_loss,
            epoch
        )


        # =====================================================
        # SAVE LAST CHECKPOINT
        # =====================================================

        save_checkpoint(
            os.path.join(
                ckpt_dir,
                "last.pt"
            ),
            model,
            optimizer,
            epoch + 1,
            best_train_loss,
            cfg
        )


        # =====================================================
        # SAVE BEST CHECKPOINT
        # =====================================================

        if train_loss < best_train_loss:

            best_train_loss = train_loss

            save_checkpoint(
                os.path.join(
                    ckpt_dir,
                    "best.pt"
                ),
                model,
                optimizer,
                epoch + 1,
                best_train_loss,
                cfg
            )

            print(
                f"[INFO] New best model saved!"
            )

            print(
                f"[INFO] Best train loss = "
                f"{best_train_loss:.4f}"
            )


    # =========================================================
    # FINISH
    # =========================================================

    writer.close()

    print(
        "\n========================================"
    )

    print(
        "TRAINING FINISHED"
    )

    print(
        "========================================"
    )

    print(
        f"Best model: "
        f"{os.path.join(ckpt_dir, 'best.pt')}"
    )

    print(
        f"Last model: "
        f"{os.path.join(ckpt_dir, 'last.pt')}"
    )


if __name__ == "__main__":
    main()