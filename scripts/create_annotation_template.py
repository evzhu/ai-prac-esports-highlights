import csv
import os

vod_id = "vod_001"
seg_dir = f"data/segments/{vod_id}"

with open("data/annotations/annotations.csv", "a", newline="") as f:
    writer = csv.writer(f)
    for fname in sorted(os.listdir(seg_dir)):
        idx = int(fname.split("_")[1].split(".")[0])
        writer.writerow([
            vod_id,
            fname.replace(".mp4",""),
            ""
        ])
