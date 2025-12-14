import os
import subprocess

SEG_LEN = 5

def segment_video(vod_path, out_dir):
    if not os.path.exists(vod_path):
        raise FileNotFoundError(f"Missing input video: {vod_path}")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i", vod_path,
        "-c", "copy",
        "-map", "0",
        "-segment_time", str(SEG_LEN),
        "-f", "segment",
        "-reset_timestamps", "1",
        os.path.join(out_dir, "seg_%05d.mp4")
    ]
    subprocess.run(cmd, check=True)

segment_video(
    "data/raw_vods/vod_002.mp4",
    "data/segments/vod_002"
)
