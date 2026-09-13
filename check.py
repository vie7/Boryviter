import csv
from pathlib import Path
from collections import defaultdict

FRAMES_ROOT = Path("wetland/frames")

max_frame_needed = defaultdict(int)
with open("dataset/bounding_boxes.csv") as f:
    for row in csv.DictReader(f, delimiter=";"):
        video_name = row["video_name"]
        frame_id = int(row["frame"])
        max_frame_needed[video_name] = max(max_frame_needed[video_name], frame_id)

for video_name, needed in sorted(max_frame_needed.items()):
    video_dir = FRAMES_ROOT / video_name
    if not video_dir.exists():
        print(f"{video_name}: MISSING FOLDER ENTIRELY")
        continue
    extracted = len(list(video_dir.glob("*.jpg")))
    if extracted <= needed:
        print(f"{video_name}: needs frame {needed}, only has {extracted} extracted")