#!/usr/bin/env python3
"""
extract_features.py

Example:
  python scripts/extract_features.py \
    --segments_csv data/annotations/segments.csv \
    --raw_dir data/raw_vods \
    --out_dir data/features \
    --fps 4 \
    --audio_sr 16000 \
    --audio_mels 64

Example only running on vod_002 and vod_003:
 python scripts/extract_features.py \
  --segments_csv data/annotations/segments.csv \
  --raw_dir data/raw_vods \
  --out_dir data/features \
  --only_vods vod_002,vod_003

segments.csv must have columns:
  vod_id,start_sec,end_sec,label

Label mapping (binary):
  H / Highlight / 1 -> 1
  anything else (including NH, N, blank) -> 0
"""

import argparse
import os
import sys
import math
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Video decode
import cv2

# Audio features
import librosa

# Video model
import torch
import torch.nn as nn
import torchvision


def _check_ffmpeg() -> None:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        raise RuntimeError("ffmpeg not found. Install ffmpeg and ensure it's on PATH.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--segments_csv", type=str, required=True)
    p.add_argument("--raw_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    # Video sampling
    p.add_argument("--fps", type=float, default=4.0, help="Sampling FPS inside each segment.")
    p.add_argument("--frame_size", type=int, default=224, help="Resize shorter side to this (square resize used).")
    p.add_argument("--max_frames", type=int, default=32, help="Max sampled frames per segment (uniform subsample).")

    # Audio features
    p.add_argument("--audio_sr", type=int, default=16000)
    p.add_argument("--audio_mels", type=int, default=64)

    # Modes
    p.add_argument("--audio_only", action="store_true")
    p.add_argument("--video_only", action="store_true")

    # Performance
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=8, help="Batch segments for video embedding (frames stacked).")

    # Restrict which vod_ids to process
    p.add_argument(
        "--only_vods",
        type=str,
        default="",
        help="Comma-separated vod_ids to process (e.g. vod_002,vod_003). If empty, process all."
    )

    # Skip if output .npz already exists
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip vods that already have an output .npz in out_dir."
    )
    p.set_defaults(skip_existing=True)  # default ON

    return p.parse_args()


def label_to_binary(label: str) -> int:
    if label is None:
        return -1
    s = str(label).strip().lower()
    if s == "":
        return -1  # unknown / unlabeled
    if s in {"h", "highlight", "1", "true", "yes"}:
        return 1
    return 0


@dataclass
class SegmentRow:
    vod_id: str
    start_sec: float
    end_sec: float
    y: int


def load_segments(path: str) -> List[SegmentRow]:
    df = pd.read_csv(path)
    required = {"vod_id", "start_sec", "end_sec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}. Need vod_id,start_sec,end_sec,label(optional).")

    if "label" not in df.columns:
        df["label"] = ""

    segs: List[SegmentRow] = []
    for _, r in df.iterrows():
        vod_id = str(r["vod_id"]).strip()
        start = float(r["start_sec"])
        end = float(r["end_sec"])
        if end <= start:
            continue
        y = label_to_binary(r.get("label", ""))
        segs.append(SegmentRow(vod_id=vod_id, start_sec=start, end_sec=end, y=y))
    return segs


def group_by_vod(segs: List[SegmentRow]) -> Dict[str, List[SegmentRow]]:
    out: Dict[str, List[SegmentRow]] = {}
    for s in segs:
        out.setdefault(s.vod_id, []).append(s)
    # sort for stable ordering
    for k in out:
        out[k].sort(key=lambda x: x.start_sec)
    return out


def ffmpeg_load_audio_chunk(
    video_path: str,
    start_sec: float,
    end_sec: float,
    sr: int
) -> np.ndarray:
    """
    Decode [start_sec, end_sec] audio to mono float32 waveform at sr using ffmpeg pipe.
    """
    duration = max(0.0, end_sec - start_sec)
    if duration <= 0:
        return np.zeros((0,), dtype=np.float32)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", str(start_sec),
        "-t", str(duration),
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-f", "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        # Return silence if decode fails (don’t crash entire job)
        return np.zeros((int(sr * duration),), dtype=np.float32)

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio


def audio_logmel_stats(audio: np.ndarray, sr: int, n_mels: int) -> np.ndarray:
    """
    Returns 2*n_mels dims: [mel_mean (n_mels), mel_std (n_mels)] => total 128 if n_mels=64.
    """
    if audio.size == 0:
        return np.zeros((2 * n_mels,), dtype=np.float32)

    # Ensure finite
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Log-mel
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=512, power=2.0)
    logS = librosa.power_to_db(S, ref=np.max)

    mel_mean = logS.mean(axis=1)
    mel_std = logS.std(axis=1)
    feat = np.concatenate([mel_mean, mel_std], axis=0).astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


def get_video_fps_and_frames(video_path: str) -> Tuple[float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not fps or fps <= 1e-6:
        fps = 30.0
    return float(fps), int(n_frames)


def read_frames_uniform(
    cap: cv2.VideoCapture,
    src_fps: float,
    start_sec: float,
    end_sec: float,
    sample_fps: float,
    frame_size: int,
    max_frames: int
) -> np.ndarray:
    """
    Returns frames as float32 tensor-like ndarray: [T, H, W, 3] in RGB, resized to (frame_size, frame_size).
    """
    duration = max(0.0, end_sec - start_sec)
    if duration <= 0:
        return np.zeros((0, frame_size, frame_size, 3), dtype=np.float32)

    # Target timestamps in seconds
    step = 1.0 / max(sample_fps, 1e-6)
    t_list = np.arange(start_sec, end_sec, step, dtype=np.float32)
    if t_list.size == 0:
        t_list = np.array([start_sec], dtype=np.float32)

    # If too many frames, uniform subsample
    if t_list.size > max_frames:
        idx = np.linspace(0, t_list.size - 1, num=max_frames).round().astype(int)
        t_list = t_list[idx]

    frames: List[np.ndarray] = []
    for t in t_list:
        frame_idx = int(round(t * src_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        frames.append(rgb.astype(np.float32) / 255.0)

    if not frames:
        return np.zeros((0, frame_size, frame_size, 3), dtype=np.float32)

    return np.stack(frames, axis=0)


class R3D18Embedder(nn.Module):
    """
    Uses pretrained torchvision r3d_18 and returns the pooled embedding before the final FC.
    Output dim: 512
    """
    def __init__(self):
        super().__init__()
        model = torchvision.models.video.r3d_18(weights=torchvision.models.video.R3D_18_Weights.DEFAULT)
        # Remove final classifier
        self.features = nn.Sequential(*list(model.children())[:-1])  # up to avgpool
        self.out_dim = 512

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, T, H, W]
        z = self.features(x)   # [B, 512, 1, 1, 1]
        z = z.flatten(1)       # [B, 512]
        return z


def frames_to_r3d_input(frames: np.ndarray, target_T: int = 16) -> np.ndarray:
    """
    r3d_18 expects fixed-ish temporal input. We’ll make exactly target_T frames:
      - if > target_T: uniform subsample
      - if < target_T: pad by repeating last frame
    Returns float32 array [3, T, H, W]
    """
    if frames.shape[0] == 0:
        # caller should handle
        raise ValueError("No frames")

    T, H, W, C = frames.shape
    if T > target_T:
        idx = np.linspace(0, T - 1, num=target_T).round().astype(int)
        frames = frames[idx]
    elif T < target_T:
        pad = np.repeat(frames[-1][None, ...], repeats=(target_T - T), axis=0)
        frames = np.concatenate([frames, pad], axis=0)

    # [T,H,W,3] -> [3,T,H,W]
    x = np.transpose(frames, (3, 0, 1, 2)).astype(np.float32)
    return x


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.audio_only and args.video_only:
        raise ValueError("Choose at most one of --audio_only / --video_only (or neither for both).")

    _check_ffmpeg()

    segs = load_segments(args.segments_csv)
    by_vod = group_by_vod(segs)

    only_set = None
    if args.only_vods.strip():
        only_set = {v.strip() for v in args.only_vods.split(",") if v.strip()}

    do_audio = not args.video_only
    do_video = not args.audio_only

    device = torch.device(args.device)

    video_model: Optional[R3D18Embedder] = None
    if do_video:
        video_model = R3D18Embedder().to(device)
        video_model.eval()

    for vod_id, vod_segs in by_vod.items():
        if only_set is not None and vod_id not in only_set:
            continue

        video_path = os.path.join(args.raw_dir, f"{vod_id}.mp4")
        if not os.path.exists(video_path):
            print(f"[WARN] Missing video for {vod_id}: {video_path} (skipping)", file=sys.stderr)
            continue

        out_path = os.path.join(args.out_dir, f"{vod_id}.npz")

        if args.skip_existing and os.path.exists(out_path):
            print(f"[SKIP] {vod_id}: already exists -> {out_path}")
            continue

        print(f"\n=== Processing {vod_id} ({len(vod_segs)} segments) ===")
        print(f"Video: {video_path}")
        print(f"Out:   {out_path}")

        # Pre-allocate outputs
        n = len(vod_segs)
        X_audio = np.zeros((n, 2 * args.audio_mels), dtype=np.float32) if do_audio else None
        X_video = np.zeros((n, 512), dtype=np.float32) if do_video else None
        y = np.array([s.y for s in vod_segs], dtype=np.int64)
        starts = np.array([s.start_sec for s in vod_segs], dtype=np.float32)
        ends = np.array([s.end_sec for s in vod_segs], dtype=np.float32)

        # Open video once for frame reads
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open {video_path}", file=sys.stderr)
            continue

        src_fps, _ = get_video_fps_and_frames(video_path)

        # We'll batch video segments for GPU efficiency
        batch_frames: List[np.ndarray] = []
        batch_indices: List[int] = []

        def flush_video_batch():
            nonlocal batch_frames, batch_indices
            if not batch_frames:
                return
            assert video_model is not None
            x = np.stack(batch_frames, axis=0)  # [B,3,T,H,W]
            xt = torch.from_numpy(x).to(device)
            with torch.no_grad():
                z = video_model(xt).detach().cpu().numpy().astype(np.float32)  # [B,512]
            for bi, seg_i in enumerate(batch_indices):
                X_video[seg_i] = z[bi]
            batch_frames = []
            batch_indices = []

        for i, seg in enumerate(tqdm(vod_segs, desc=f"{vod_id} segments")):
            # Audio
            if do_audio:
                audio = ffmpeg_load_audio_chunk(video_path, seg.start_sec, seg.end_sec, args.audio_sr)
                X_audio[i] = audio_logmel_stats(audio, args.audio_sr, args.audio_mels)

            # Video frames -> embedding
            if do_video:
                frames = read_frames_uniform(
                    cap=cap,
                    src_fps=src_fps,
                    start_sec=seg.start_sec,
                    end_sec=seg.end_sec,
                    sample_fps=args.fps,
                    frame_size=args.frame_size,
                    max_frames=args.max_frames
                )
                if frames.shape[0] == 0:
                    # If decode fails, just leave zeros
                    X_video[i] = 0.0
                else:
                    x = frames_to_r3d_input(frames, target_T=16)  # [3,T,H,W]
                    batch_frames.append(x)
                    batch_indices.append(i)
                    if len(batch_frames) >= args.batch_size:
                        flush_video_batch()

        # Flush leftover batch
        if do_video:
            flush_video_batch()

        cap.release()

        # Save
        save_kwargs = {
            "y": y,
            "start_sec": starts,
            "end_sec": ends,
        }
        if do_audio:
            save_kwargs["X_audio"] = X_audio
        if do_video:
            save_kwargs["X_video"] = X_video

        np.savez_compressed(out_path, **save_kwargs)
        print(f"Saved {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
