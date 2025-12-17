#!/usr/bin/env python3
"""
train_classifier_cross_attn.py

Train a binary highlight classifier from per-segment embeddings saved as .npz.

Expected each .npz to contain:
  - y: int array shape [N] (0/1)
  - start_sec, end_sec: float arrays shape [N] (optional)
  - X_video: float array shape [N, Dv] (optional if audio-only)
  - X_audio: float array shape [N, Da] (optional if video-only)

Models:
  - mlp: MLP on per-segment features (concat)
  - logreg: logistic regression on per-segment features (concat)
  - cross_attn: cross-modal attention fusion (video attends to audio) on a per-VOD sequence

Example:
  py scripts/train_classifier_cross_attn.py \
    --features_dir data/features \
    --holdout_vod vod_003 \
    --model_type cross_attn \
    --out_path data/models/model_cross_attn.pt
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Sklearn for logreg baseline + metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class VodData:
    vod_id: str
    y: np.ndarray                 # [N]
    X: Optional[np.ndarray] = None  # [N, D] for concat models
    X_video: Optional[np.ndarray] = None  # [N, Dv] for cross_attn
    X_audio: Optional[np.ndarray] = None  # [N, Da] for cross_attn


# -----------------------------
# Feature selection + loading
# -----------------------------
def choose_feature_flags(args: argparse.Namespace, sample_npz: str) -> Tuple[bool, bool]:
    """
    Returns (use_audio, use_video) based on CLI flags and/or availability in sample_npz.
    """
    if args.audio_only and args.video_only:
        raise ValueError("Choose only one of --audio_only/--video_only")

    if args.audio_only:
        return True, False
    if args.video_only:
        return False, True

    # If user explicitly requested use_audio/use_video, obey
    if args.use_audio or args.use_video:
        return bool(args.use_audio), bool(args.use_video)

    # Otherwise auto-detect based on keys in one file
    d = np.load(sample_npz)
    has_a = "X_audio" in d.files
    has_v = "X_video" in d.files
    if has_a and has_v:
        return True, True
    if has_a:
        return True, False
    if has_v:
        return False, True
    raise ValueError(f"{sample_npz} has neither X_audio nor X_video.")


def load_npz_concat(path: str, vod_id: str, want_audio: bool, want_video: bool) -> VodData:
    """
    Load a .npz into concatenated X for per-segment models.
    """
    d = np.load(path)
    if "y" not in d.files:
        raise ValueError(f"{path} missing key 'y'")

    y = d["y"].astype(np.int64)
    parts: List[np.ndarray] = []

    if want_video:
        if "X_video" not in d.files:
            raise ValueError(f"{path}: want_video=True but no X_video found.")
        parts.append(d["X_video"].astype(np.float32))

    if want_audio:
        if "X_audio" not in d.files:
            raise ValueError(f"{path}: want_audio=True but no X_audio found.")
        parts.append(d["X_audio"].astype(np.float32))

    if not parts:
        raise ValueError("No features selected (want_audio and want_video are both False).")

    X = np.concatenate(parts, axis=1).astype(np.float32)
    return VodData(vod_id=vod_id, y=y, X=X)


def load_npz_split(path: str, vod_id: str) -> VodData:
    """
    Load a .npz into separate (X_video, X_audio) for cross-attn fusion.
    Requires both to exist.
    """
    d = np.load(path)
    if "y" not in d.files:
        raise ValueError(f"{path} missing key 'y'")
    if "X_video" not in d.files or "X_audio" not in d.files:
        raise ValueError(f"{path}: cross_attn requires BOTH X_video and X_audio.")

    y = d["y"].astype(np.int64)
    Xv = d["X_video"].astype(np.float32)
    Xa = d["X_audio"].astype(np.float32)
    if Xv.shape[0] != Xa.shape[0]:
        raise ValueError(f"{path}: X_video and X_audio have different lengths: {Xv.shape[0]} vs {Xa.shape[0]}")
    return VodData(vod_id=vod_id, y=y, X_video=Xv, X_audio=Xa)


def standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score using train mean/std only.
    Returns standardized train/test and (mean, std).
    """
    mu = X_train.mean(axis=0, keepdims=True)
    sig = X_train.std(axis=0, keepdims=True)
    sig = np.where(sig < 1e-8, 1.0, sig)
    Xtr = (X_train - mu) / sig
    Xte = (X_test - mu) / sig
    return Xtr.astype(np.float32), Xte.astype(np.float32), mu.astype(np.float32), sig.astype(np.float32)


def standardize_with_stats(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    sig = np.where(sig < 1e-8, 1.0, sig)
    return ((X - mu) / sig).astype(np.float32)


# -----------------------------
# Torch dataset + models
# -----------------------------
class NPArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class VODSequenceDataset(Dataset):
    """
    Each item is one VOD as a variable-length sequence.
    Returns (V, A, y, length) with shapes:
      V: [N, Dv], A: [N, Da], y: [N]
    """
    def __init__(self, vods: List[VodData]):
        self.vods = vods

    def __len__(self) -> int:
        return len(self.vods)

    def __getitem__(self, idx: int):
        vd = self.vods[idx]
        assert vd.X_video is not None and vd.X_audio is not None
        V = torch.from_numpy(vd.X_video)          # [N, Dv]
        A = torch.from_numpy(vd.X_audio)          # [N, Da]
        y = torch.from_numpy(vd.y).float()        # [N]
        n = V.shape[0]
        return V, A, y, n


def collate_vod_sequences(batch):
    """
    Pads variable-length VOD sequences to [B, Lmax, D*].
    Returns:
      V_pad: [B, L, Dv]
      A_pad: [B, L, Da]
      y_pad: [B, L] (padded with -1)
      key_padding_mask: [B, L] True where padded
    """
    V_list, A_list, y_list, n_list = zip(*batch)
    B = len(V_list)
    L = max(int(n) for n in n_list)
    Dv = V_list[0].shape[1]
    Da = A_list[0].shape[1]

    V_pad = torch.zeros((B, L, Dv), dtype=torch.float32)
    A_pad = torch.zeros((B, L, Da), dtype=torch.float32)
    y_pad = torch.full((B, L), -1.0, dtype=torch.float32)
    key_padding_mask = torch.ones((B, L), dtype=torch.bool)  # True = ignore

    for i, (V, A, y, n) in enumerate(zip(V_list, A_list, y_list, n_list)):
        n = int(n)
        V_pad[i, :n] = V
        A_pad[i, :n] = A
        y_pad[i, :n] = y
        key_padding_mask[i, :n] = False

    return V_pad, A_pad, y_pad, key_padding_mask


class MLPBinary(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CrossModalFusion(nn.Module):
    """
    Cross-modal attention: video queries attend to audio keys/values.
    Input:
      V: [B, L, Dv]
      A: [B, L, Da]
    Output:
      logits: [B, L]
    """
    def __init__(self, dv: int, da: int, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dv = dv
        self.da = da
        self.d_model = d_model
        self.n_heads = n_heads

        self.v_proj = nn.Linear(dv, d_model)
        self.a_proj = nn.Linear(da, d_model)

        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, 1)

    def forward(self, V: torch.Tensor, A: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        V = self.v_proj(V)  # [B, L, d]
        A = self.a_proj(A)  # [B, L, d]

        Z, _ = self.cross_attn(query=V, key=A, value=A, key_padding_mask=key_padding_mask)  # [B, L, d]
        X = self.ln1(V + Z)
        X = self.ln2(X + self.ffn(X))
        logits = self.head(X).squeeze(-1)  # [B, L]
        return logits


# -----------------------------
# Metrics helpers
# -----------------------------
@torch.no_grad()
def eval_mlp(model: nn.Module, X: np.ndarray, y: np.ndarray, device: str) -> Dict[str, float]:
    model.eval()
    xb = torch.from_numpy(X).to(device)
    logits = model(xb).detach().cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_true = y.astype(np.int64)
    y_pred = (probs >= 0.5).astype(np.int64)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    ap = average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "pr_auc": float(ap)}


@torch.no_grad()
def eval_cross_attn(
    model: nn.Module,
    vods: List[VodData],
    device: str,
    threshold: float = 0.5,
) -> Dict[str, object]:
    model.eval()
    all_probs: List[np.ndarray] = []
    all_y: List[np.ndarray] = []

    for vd in vods:
        assert vd.X_video is not None and vd.X_audio is not None
        V = torch.from_numpy(vd.X_video).unsqueeze(0).to(device)  # [1, N, Dv]
        A = torch.from_numpy(vd.X_audio).unsqueeze(0).to(device)  # [1, N, Da]
        logits = model(V, A, key_padding_mask=None).squeeze(0).detach().cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        all_probs.append(probs.astype(np.float32))
        all_y.append(vd.y.astype(np.int64))

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_y, axis=0)

    y_pred = (probs >= threshold).astype(np.int64)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    pr_auc = (
        average_precision_score(y_true, probs)
        if len(np.unique(y_true)) > 1
        else float("nan")
    )
    roc_auc = (
        roc_auc_score(y_true, probs)
        if len(np.unique(y_true)) > 1
        else float("nan")
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "threshold": float(threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm,
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True, help="Directory containing per-VOD .npz feature files")
    ap.add_argument("--holdout_vod", required=True, help="VOD ID to hold out for testing (e.g., vod_003)")
    ap.add_argument("--out_path", required=True, help="Where to save model checkpoint .pt")

    ap.add_argument("--model_type", default="mlp", choices=["mlp", "logreg", "cross_attn"])
    ap.add_argument("--audio_only", action="store_true")
    ap.add_argument("--video_only", action="store_true")
    ap.add_argument("--use_audio", action="store_true", help="Force include audio when both exist")
    ap.add_argument("--use_video", action="store_true", help="Force include video when both exist")

    # MLP hyperparams
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)

    # Cross-attn hyperparams
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)

    # Optim
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256, help="For mlp/logreg this is segments; for cross_attn this is VODs per batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")

    args = ap.parse_args()

    # Collect .npz
    npz_paths = sorted(
        [os.path.join(args.features_dir, f) for f in os.listdir(args.features_dir) if f.endswith(".npz")]
    )
    if not npz_paths:
        raise FileNotFoundError(f"No .npz found in {args.features_dir}")

    # Special handling: cross_attn requires both audio+video
    if args.model_type == "cross_attn":
        if args.audio_only or args.video_only:
            raise ValueError("cross_attn requires BOTH audio and video. Remove --audio_only/--video_only.")
        use_audio, use_video = True, True
    else:
        use_audio, use_video = choose_feature_flags(args, npz_paths[0])

    holdout_path = os.path.join(args.features_dir, f"{args.holdout_vod}.npz")
    if not os.path.exists(holdout_path):
        raise FileNotFoundError(
            f"Holdout file not found: {holdout_path}\n"
            f"Existing: {[os.path.basename(p) for p in npz_paths]}"
        )

    # Load all VODs
    train_vods: List[VodData] = []
    test_vod: Optional[VodData] = None

    for p in npz_paths:
        vod_id = os.path.splitext(os.path.basename(p))[0]
        if args.model_type == "cross_attn":
            vd = load_npz_split(p, vod_id=vod_id)
        else:
            vd = load_npz_concat(p, vod_id=vod_id, want_audio=use_audio, want_video=use_video)

        if vod_id == args.holdout_vod:
            test_vod = vd
        else:
            train_vods.append(vd)

    if test_vod is None:
        # If holdout wasn't in glob, load directly
        test_vod = (
            load_npz_split(holdout_path, vod_id=args.holdout_vod)
            if args.model_type == "cross_attn"
            else load_npz_concat(holdout_path, vod_id=args.holdout_vod, want_audio=use_audio, want_video=use_video)
        )

    if not train_vods:
        raise ValueError("No training VODs found (everything is holdout?). Add more .npz or change --holdout_vod.")

    # -----------------------------
    # Train per-segment baselines
    # -----------------------------
    if args.model_type in ("mlp", "logreg"):
        # Flatten across VODs
        X_train = np.concatenate([vd.X for vd in train_vods if vd.X is not None], axis=0)
        y_train = np.concatenate([vd.y for vd in train_vods], axis=0).astype(np.int64)

        assert test_vod.X is not None
        X_test = test_vod.X
        y_test = test_vod.y.astype(np.int64)

        # Standardize
        X_train_s, X_test_s, mu, sig = standardize_train_test(X_train, X_test)

        # Print dataset summary like your other script
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        test_pos = int((y_test == 1).sum())
        test_neg = int((y_test == 0).sum())
        print(f"Train segments: {len(y_train)}  positives={pos}  negatives={neg}  pos_rate={pos / max(len(y_train), 1):.4f}")
        print(f"Test  segments: {len(y_test)}  positives={test_pos}  negatives={test_neg}  pos_rate={test_pos / max(len(y_test), 1):.4f}")
        print(f"Using features: audio={use_audio}, video={use_video}, dim={int(X_train_s.shape[1])}")
        print(f"Model: {args.model_type}")

        # Train
        if args.model_type == "logreg":
            clf = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
            )
            clf.fit(X_train_s, y_train)

            probs = clf.predict_proba(X_test_s)[:, 1]
            y_pred = (probs >= 0.5).astype(np.int64)
            p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
            ap_score = average_precision_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan")

            print(f"[HOLDOUT {args.holdout_vod}] precision={p:.3f} recall={r:.3f} f1={f1:.3f} pr_auc={ap_score:.3f}")

            ckpt = {
                "model_type": "logreg",
                "sk_model": clf,
                "feature_mean": mu.squeeze(0),
                "feature_std": sig.squeeze(0),
                "feature_dim": int(X_train_s.shape[1]),
                "config": {
                    "use_audio": bool(use_audio),
                    "use_video": bool(use_video),
                },
            }
            torch.save(ckpt, args.out_path)
            print(f"Saved checkpoint: {args.out_path}")
            return

        # Torch MLP
        device = args.device
        in_dim = int(X_train_s.shape[1])
        model = MLPBinary(in_dim=in_dim, hidden=args.hidden, dropout=args.dropout).to(device)

        # Imbalance weights
        pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_dl = DataLoader(
            NPArrayDataset(X_train_s, y_train.astype(np.float32)),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            for xb, yb in train_dl:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * xb.shape[0]

            train_loss = running / max(len(train_dl.dataset), 1)

            metrics = eval_mlp(model, X_test_s, y_test, device=device)
            print(
                f"Epoch {epoch:02d}/{args.epochs}  "
                f"loss={train_loss:.4f}  "
                f"test_f1={metrics['f1']:.3f}  prec={metrics['precision']:.3f}  rec={metrics['recall']:.3f}  "
                f"pr_auc={metrics['pr_auc']:.3f}"
            )

        final_metrics = eval_mlp(model, X_test_s, y_test, device=device)
        print("FINAL TEST METRICS:")
        print(json.dumps({"threshold": 0.5, **final_metrics}, indent=2))

        ckpt = {
            "model_type": "mlp",
            "state_dict": model.state_dict(),
            "feature_mean": mu.squeeze(0),
            "feature_std": sig.squeeze(0),
            "feature_dim": int(in_dim),
            "config": {
                "hidden": int(args.hidden),
                "dropout": float(args.dropout),
                "use_audio": bool(use_audio),
                "use_video": bool(use_video),
            },
        }
        torch.save(ckpt, args.out_path)
        print(f"Saved checkpoint: {args.out_path}")
        return

    # -----------------------------
    # Train cross-modal attention fusion
    # -----------------------------
    device = args.device
    assert test_vod is not None and test_vod.X_video is not None and test_vod.X_audio is not None

    # Build scaler on concatenated features across all TRAIN segments
    X_train_concat = np.concatenate(
        [np.concatenate([vd.X_video, vd.X_audio], axis=1) for vd in train_vods], axis=0
    )  # [SumN, Dv+Da]
    X_test_concat = np.concatenate([test_vod.X_video, test_vod.X_audio], axis=1)

    X_train_s, X_test_s, mu, sig = standardize_train_test(X_train_concat, X_test_concat)

    # Split standardized back into video/audio per VOD
    dv = int(train_vods[0].X_video.shape[1])
    da = int(train_vods[0].X_audio.shape[1])

    def apply_split_standardization(vd: VodData, mu_: np.ndarray, sig_: np.ndarray) -> VodData:
        Xc = np.concatenate([vd.X_video, vd.X_audio], axis=1)
        Xcs = standardize_with_stats(Xc, mu_, sig_)
        vd.X_video = Xcs[:, :dv]
        vd.X_audio = Xcs[:, dv:dv + da]
        return vd

    train_vods = [apply_split_standardization(vd, mu, sig) for vd in train_vods]
    test_vod = apply_split_standardization(test_vod, mu, sig)

    # ---- dataset summary (cross-attn) ----
    train_y = np.concatenate([vd.y for vd in train_vods]).astype(np.int64)
    test_y = test_vod.y.astype(np.int64)

    train_pos = int((train_y == 1).sum())
    train_neg = int((train_y == 0).sum())
    test_pos = int((test_y == 1).sum())
    test_neg = int((test_y == 0).sum())

    print(f"Train segments: {len(train_y)}  positives={train_pos}  negatives={train_neg}  pos_rate={train_pos / max(len(train_y), 1):.4f}")
    print(f"Test  segments: {len(test_y)}  positives={test_pos}  negatives={test_neg}  pos_rate={test_pos / max(len(test_y), 1):.4f}")
    print(f"Using features: audio=True, video=True, dim={dv + da}  (dv={dv}, da={da})")
    print(f"Model: cross_attn  d_model={args.d_model}  n_heads={args.n_heads}  dropout={args.dropout}")

    # Build model
    model = CrossModalFusion(dv=dv, da=da, d_model=args.d_model, n_heads=args.n_heads, dropout=args.dropout).to(device)

    # Compute pos_weight over TRAIN tokens
    pos_weight = torch.tensor([train_neg / max(train_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_dl = DataLoader(
        VODSequenceDataset(train_vods),
        batch_size=min(args.batch_size, len(train_vods)),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_vod_sequences,
    )

    # For reporting during training
    holdout_list = [test_vod]

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_tokens = 0

        for V_pad, A_pad, y_pad, key_padding_mask in train_dl:
            V_pad = V_pad.to(device)
            A_pad = A_pad.to(device)
            y_pad = y_pad.to(device)
            key_padding_mask = key_padding_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(V_pad, A_pad, key_padding_mask=key_padding_mask)  # [B, L]

            valid = ~key_padding_mask  # [B, L]
            y_valid = y_pad[valid]
            logits_valid = logits[valid]

            loss = criterion(logits_valid, y_valid)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * int(y_valid.numel())
            n_tokens += int(y_valid.numel())

        train_loss = running_loss / max(n_tokens, 1)

        metrics = eval_cross_attn(model, holdout_list, device=device, threshold=0.5)
        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"loss={train_loss:.4f}  "
            f"test_f1={metrics['f1']:.3f}  prec={metrics['precision']:.3f}  rec={metrics['recall']:.3f}  "
            f"pr_auc={metrics['pr_auc']:.3f}  roc_auc={metrics['roc_auc']:.3f}"
        )

    final_metrics = eval_cross_attn(model, holdout_list, device=device, threshold=0.5)
    print("FINAL TEST METRICS:")
    print(json.dumps(final_metrics, indent=2))

    ckpt = {
        "model_type": "cross_attn",
        "state_dict": model.state_dict(),
        # scaler stats over concatenated dims
        "feature_mean": mu.squeeze(0),
        "feature_std": sig.squeeze(0),
        "feature_dim": int(dv + da),
        "config": {
            "dv": dv,
            "da": da,
            "d_model": int(args.d_model),
            "n_heads": int(args.n_heads),
            "dropout": float(args.dropout),
            "use_audio": True,
            "use_video": True,
        },
    }
    torch.save(ckpt, args.out_path)
    print(f"Saved checkpoint: {args.out_path}")


if __name__ == "__main__":
    main()
