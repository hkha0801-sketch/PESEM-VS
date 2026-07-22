import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import load_config, set_seed, get_device, save_checkpoint, load_checkpoint
from dataset import SpeechEnhancementDataset, collate_fn
from model import build_model
from losses import get_loss_fn
from metrics import evaluate_batch


def run_epoch(model, loader, loss_fn, optimizer, device, grad_clip, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for noisy, clean in tqdm(loader, desc="train" if train else "val", leave=False):
            noisy, clean = noisy.to(device), clean.to(device)
            est = model(noisy)
            min_len = min(est.shape[-1], clean.shape[-1])
            loss = loss_fn(est[..., :min_len], clean[..., :min_len])

            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate_test_set(model, loader, device, sample_rate):
    model.eval()
    all_metrics = {}
    n = 0
    with torch.no_grad():
        for noisy, clean in tqdm(loader, desc="test", leave=False):
            noisy = noisy.to(device)
            est = model(noisy).cpu()
            min_len = min(est.shape[-1], clean.shape[-1])
            m = evaluate_batch(est[..., :min_len], clean[..., :min_len], sr=sample_rate)
            for k, v in m.items():
                all_metrics[k] = all_metrics.get(k, 0.0) + v
            n += 1
    return {k: v / max(n, 1) for k, v in all_metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"].get("seed", 42))
    device = get_device(cfg["train"].get("device", "auto"))
    print(f"[INFO] device = {device}")

    data_cfg = cfg["data"]
    train_ds = SpeechEnhancementDataset(
        data_cfg["train_csv"], sample_rate=data_cfg["sample_rate"],
        segment_seconds=data_cfg["segment_seconds"], train=True,
    )
    val_ds = SpeechEnhancementDataset(
        data_cfg["val_csv"], sample_rate=data_cfg["sample_rate"],
        segment_seconds=data_cfg["segment_seconds"], train=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"], collate_fn=collate_fn,
    )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 0.0)
    )
    loss_fn = get_loss_fn(cfg["train"]["loss"], stft_cfg=cfg["stft"])

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        start_epoch, best_val = load_checkpoint(args.resume, model, optimizer, map_location=device)
        print(f"[INFO] resumed from {args.resume}, epoch={start_epoch}, best_val={best_val:.4f}")

    ckpt_dir = cfg["train"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(cfg["train"].get("log_dir", "runs"))

    patience = cfg["train"].get("early_stop_patience", 15)
    epochs_no_improve = 0

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device,
                                cfg["train"].get("grad_clip"), train=True)
        val_loss = run_epoch(model, val_loader, loss_fn, optimizer, device,
                              cfg["train"].get("grad_clip"), train=False)

        print(f"[Epoch {epoch+1}/{cfg['train']['epochs']}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)

        save_checkpoint(os.path.join(ckpt_dir, "last.pt"), model, optimizer, epoch + 1, best_val, cfg)

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), model, optimizer, epoch + 1, best_val, cfg)
            print(f"[INFO] new best model saved, val_loss={val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[INFO] early stopping tại epoch {epoch+1} (không cải thiện {patience} epoch)")
                break

    # Đánh giá tập test nếu có
    test_csv = data_cfg.get("test_csv")
    if test_csv and os.path.exists(test_csv):
        print("[INFO] đánh giá trên tập test...")
        test_ds = SpeechEnhancementDataset(
            test_csv, sample_rate=data_cfg["sample_rate"], segment_seconds=None, train=False,
        )
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
        load_checkpoint(os.path.join(ckpt_dir, "best.pt"), model, map_location=device)
        metrics = evaluate_test_set(model, test_loader, device, data_cfg["sample_rate"])
        print(f"[TEST METRICS] {metrics}")

    writer.close()


if __name__ == "__main__":
    main()
