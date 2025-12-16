#!/usr/bin/env python3
"""
postprocess_intervals.py

Convert per-segment predictions into final clip intervals.

Input preds.csv columns:
  vod_id,start_sec,end_sec,prob

Output clips.csv columns:
  vod_id,clip_start,clip_end,avg_prob,max_prob,num_segments

Example:
  python scripts/postprocess_intervals.py \
    --preds_csv data/preds/preds.csv \
    --threshold 0.5 \
    --merge_gap 5 \
    --context 3 \
    --raw_dir data/raw_vods \
    --out_csv data/output/clips.csv

Notes:
- If --raw_dir is provided and ffprobe is available, clip times are clamped to video duration.
- Otherwise, clipping still works, but times are not duration-clamped.
"""

import argparse
import csv
import os
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Interval:
    start: float
    end: float
    probs: List[float]  # probs of segments merged into this interval

    @property
    def avg_prob(self) -> float:
        return float(np.mean(self.probs)) if self.probs else 0.0

    @property
    def max_prob(self) -> float:
        return float(np.max(self.probs)) if self.probs else 0.0

    @property
    def num_segments(self) -> int:
        return int(len(self.probs))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preds_csv", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)

    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--merge_gap", type=float, default=5.0, help="Merge if next.start - cur.end <= merge_gap")
    p.add_argument("--context", type=float, default=3.0, help="Expand each merged interval by ±context seconds")

    # Optional duration clamping
    p.add_argument("--raw_dir", type=str, default="", help="If provided, clamp clips to VOD duration via ffprobe")
    p.add_argument("--video_ext", type=str, default=".mp4")

    # Filters / sorting
    p.add_argument("--min_clip_len", type=float, default=0.0, help="Drop clips shorter than this many seconds")
    p.add_argument("--top_k", type=int, default=0, help="If >0, keep only top_k clips per VOD by max_prob")

    return p.parse_args()


def get_duration_ffprobe(video_path: str) -> Optional[float]:
    """
    Returns duration in seconds using ffprobe, or None if unavailable.
    """
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
        dur = float(out)
        if dur > 0:
            return dur
    except Exception:
        return None
    return None


def merge_intervals(intervals: List[Interval], merge_gap: float) -> List[Interval]:
    """
    Merge overlapping or near-adjacent intervals (gap <= merge_gap).
    intervals must be sorted by start.
    """
    if not intervals:
        return []

    merged: List[Interval] = []
    cur = intervals[0]

    for nxt in intervals[1:]:
        gap = nxt.start - cur.end
        if gap <= merge_gap:
            # merge
            cur.end = max(cur.end, nxt.end)
            cur.probs.extend(nxt.probs)
        else:
            merged.append(cur)
            cur = nxt

    merged.append(cur)
    return merged


def clamp_interval(start: float, end: float, lo: float, hi: float) -> Tuple[float, float]:
    s = max(lo, start)
    e = min(hi, end)
    if e < s:
        e = s
    return s, e


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    df = pd.read_csv(args.preds_csv)
    required = {"vod_id", "start_sec", "end_sec", "prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{args.preds_csv} missing columns: {sorted(missing)}")

    # Filter by threshold
    df = df[df["prob"] >= args.threshold].copy()
    if df.empty:
        # Still write empty output (useful for debugging)
        pd.DataFrame(columns=[
            "vod_id", "clip_start", "clip_end", "avg_prob", "max_prob", "num_segments"
        ]).to_csv(args.out_csv, index=False)
        print(f"No segments >= threshold={args.threshold}. Wrote empty {args.out_csv}")
        return

    # Ensure numeric
    df["start_sec"] = df["start_sec"].astype(float)
    df["end_sec"] = df["end_sec"].astype(float)
    df["prob"] = df["prob"].astype(float)

    out_rows = []

    for vod_id, g in df.groupby("vod_id"):
        g = g.sort_values("start_sec")

        # Build initial intervals from kept segments
        intervals = [
            Interval(float(r.start_sec), float(r.end_sec), [float(r.prob)])
            for r in g.itertuples(index=False)
            if float(r.end_sec) > float(r.start_sec)
        ]

        if not intervals:
            continue

        # 1) merge adjacent/overlapping
        merged = merge_intervals(intervals, merge_gap=args.merge_gap)

        # Optional duration clamp
        duration = None
        if args.raw_dir:
            video_path = os.path.join(args.raw_dir, f"{vod_id}{args.video_ext}")
            if os.path.exists(video_path):
                duration = get_duration_ffprobe(video_path)

        # 2) add context and clamp
        expanded: List[Interval] = []
        for iv in merged:
            s = iv.start - args.context
            e = iv.end + args.context
            if duration is not None:
                s, e = clamp_interval(s, e, 0.0, duration)
            else:
                s = max(0.0, s)
            expanded.append(Interval(s, e, probs=list(iv.probs)))

        # 3) merge again after expansion (handles overlaps from context)
        expanded.sort(key=lambda x: x.start)
        final = merge_intervals(expanded, merge_gap=0.0)

        # 4) filter too-short
        final = [iv for iv in final if (iv.end - iv.start) >= args.min_clip_len]

        # 5) optionally keep top-k per VOD by max_prob
        if args.top_k and args.top_k > 0:
            final = sorted(final, key=lambda x: x.max_prob, reverse=True)[: args.top_k]
            final.sort(key=lambda x: x.start)

        for iv in final:
            out_rows.append({
                "vod_id": vod_id,
                "clip_start": round(iv.start, 3),
                "clip_end": round(iv.end, 3),
                "avg_prob": round(iv.avg_prob, 6),
                "max_prob": round(iv.max_prob, 6),
                "num_segments": iv.num_segments,
            })

    out_df = pd.DataFrame(out_rows, columns=[
        "vod_id", "clip_start", "clip_end", "avg_prob", "max_prob", "num_segments"
    ])
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} clips to {args.out_csv}")
    if len(out_df) > 0:
        print(out_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
