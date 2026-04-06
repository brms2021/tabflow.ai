"""Training script for guitar tablature models.

Usage:
    python -m training.train --model pitch --data-dir data/processed
    python -m training.train --model fret --data-dir data/processed
    python -m training.train --model technique --data-dir data/processed
"""

import argparse
import glob
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .dataset import FretAssignmentDataset, PitchDetectionDataset, TechniqueDataset
from .models import FretNet, GuitarPitchNet, TechniqueNet

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def train_pitch_model(data_dir: str, epochs: int = 50, lr: float = 1e-3, batch_size: int = 64) -> None:
    """Train guitar pitch detection model."""
    logger.info("=== Training GuitarPitchNet ===")

    # Find aligned data + audio pairs
    aligned_files = sorted(glob.glob(f"{data_dir}/**/aligned.json", recursive=True))
    audio_files = []
    valid_aligned = []
    for af in aligned_files:
        # Find corresponding audio
        track_dir = Path(af).parent.name
        audio_candidates = glob.glob(f"data/raw/{track_dir}/*120bpm*.mp3") + glob.glob(
            f"data/raw/{track_dir}/audio.mp3"
        )
        if audio_candidates:
            audio_files.append(audio_candidates[0])
            valid_aligned.append(af)

    if not valid_aligned:
        logger.error("No aligned data found in %s", data_dir)
        return

    logger.info("Found %d aligned tracks", len(valid_aligned))

    dataset = PitchDetectionDataset(valid_aligned, audio_files)
    logger.info("Dataset: %d samples", len(dataset))

    if len(dataset) < 10:
        logger.error("Not enough data to train (need at least 10 samples)")
        return

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = GuitarPitchNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    logger.info("Model: %d parameters", sum(p.numel() for p in model.parameters()))

    best_val_loss = float("inf")
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("Epoch %d/%d — train_loss=%.4f val_loss=%.4f", epoch + 1, epochs, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/guitar_pitch_net.pt")

    logger.info("Best val_loss: %.4f — saved to models/guitar_pitch_net.pt", best_val_loss)


def train_fret_model(data_dir: str, epochs: int = 100, lr: float = 1e-3, batch_size: int = 32) -> None:
    """Train neural fret assignment model."""
    logger.info("=== Training FretNet ===")

    gt_files = sorted(glob.glob(f"{data_dir}/**/*.ground_truth.json", recursive=True))
    if not gt_files:
        logger.error("No ground truth files found in %s", data_dir)
        return

    logger.info("Found %d ground truth files", len(gt_files))

    dataset = FretAssignmentDataset(gt_files)
    logger.info("Dataset: %d sequences", len(dataset))

    if len(dataset) < 5:
        logger.error("Not enough data to train")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = FretNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    string_criterion = nn.CrossEntropyLoss()
    fret_criterion = nn.CrossEntropyLoss()

    logger.info("Model: %d parameters", sum(p.numel() for p in model.parameters()))

    Path("models").mkdir(exist_ok=True)
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            string_logits, fret_logits = model(batch_x)

            # batch_y: (batch, seq_len, 2) — string, fret
            string_targets = batch_y[:, :, 0]  # (batch, seq_len)
            fret_targets = batch_y[:, :, 1]  # (batch, seq_len)

            # Reshape for cross-entropy: (batch*seq_len, classes) vs (batch*seq_len,)
            s_loss = string_criterion(string_logits.reshape(-1, 6), string_targets.reshape(-1))
            f_loss = fret_criterion(fret_logits.reshape(-1, 25), fret_targets.reshape(-1))
            loss = s_loss + f_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct_strings = 0
        correct_frets = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                string_logits, fret_logits = model(batch_x)
                string_targets = batch_y[:, :, 0]
                fret_targets = batch_y[:, :, 1]

                s_loss = string_criterion(string_logits.reshape(-1, 6), string_targets.reshape(-1))
                f_loss = fret_criterion(fret_logits.reshape(-1, 25), fret_targets.reshape(-1))
                val_loss += (s_loss + f_loss).item()

                # Accuracy
                pred_strings = string_logits.argmax(dim=-1)
                pred_frets = fret_logits.argmax(dim=-1)
                correct_strings += (pred_strings == string_targets).sum().item()
                correct_frets += (pred_frets == fret_targets).sum().item()
                total += string_targets.numel()

        val_loss /= max(len(val_loader), 1)
        str_acc = correct_strings / max(total, 1)
        fret_acc = correct_frets / max(total, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "Epoch %d/%d — train=%.4f val=%.4f str_acc=%.1f%% fret_acc=%.1f%%",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                str_acc * 100,
                fret_acc * 100,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/fret_net.pt")

    logger.info("Best val_loss: %.4f — saved to models/fret_net.pt", best_val_loss)


def train_technique_model(data_dir: str, epochs: int = 50, lr: float = 1e-3, batch_size: int = 32) -> None:
    """Train technique classification model."""
    logger.info("=== Training TechniqueNet ===")

    # Find aligned data with technique labels
    aligned_files = sorted(glob.glob(f"{data_dir}/**/aligned.json", recursive=True))
    audio_files = []
    valid_aligned = []
    for af in aligned_files:
        track_dir = Path(af).parent.name
        audio_candidates = glob.glob(f"data/raw/{track_dir}/*120bpm*.mp3") + glob.glob(
            f"data/raw/{track_dir}/audio.mp3"
        )
        if audio_candidates:
            audio_files.append(audio_candidates[0])
            valid_aligned.append(af)

    if not valid_aligned:
        logger.error("No aligned data found")
        return

    dataset = TechniqueDataset(valid_aligned, audio_files)
    logger.info("Dataset: %d samples", len(dataset))

    # Check class distribution
    from collections import Counter

    labels = [dataset[i][1] for i in range(len(dataset))]
    dist = Counter(labels)
    logger.info("Class distribution: %s", dict(dist))

    if len(dataset) < 10:
        logger.error("Not enough data")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = TechniqueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Class weights for imbalanced data
    class_counts = torch.zeros(8)
    for label in labels:
        class_counts[label] += 1
    weights = 1.0 / (class_counts + 1)
    criterion = nn.CrossEntropyLoss(weight=weights)

    logger.info("Model: %d parameters", sum(p.numel() for p in model.parameters()))

    Path("models").mkdir(exist_ok=True)
    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
                correct += (pred.argmax(dim=1) == batch_y).sum().item()
                total += len(batch_y)
        val_loss /= max(len(val_loader), 1)
        acc = correct / max(total, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "Epoch %d/%d — train=%.4f val=%.4f acc=%.1f%%", epoch + 1, epochs, train_loss, val_loss, acc * 100
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/technique_net.pt")

    logger.info("Best val_loss: %.4f — saved to models/technique_net.pt", best_val_loss)


def main():
    parser = argparse.ArgumentParser(description="Train guitar tablature models")
    parser.add_argument("--model", choices=["pitch", "fret", "technique"], required=True)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)

    if args.model == "pitch":
        train_pitch_model(args.data_dir, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    elif args.model == "fret":
        train_fret_model(args.data_dir, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    elif args.model == "technique":
        train_technique_model(args.data_dir, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
