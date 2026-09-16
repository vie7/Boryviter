import csv
import os
from collections import defaultdict

import cv2

CSV_PATH = "wetland/bounding_boxes.csv"
VIDEOS_DIR = "wetland/videos"
FRAMES_ROOT = "wetland/frames"

"""
Extract frame image from the source video at each annotated frame number.
"""

def get_needed_frames(csv_path):
    """video_name -> set of frame numbers that have annotations."""
    needed = defaultdict(set)
    with open(csv_path) as f:
        for row in csv.DictReader(f, delimiter=";"):
            needed[row["video_name"]].add(int(row["frame"]))
    return needed


def extract_video(video_name, frame_numbers, videos_dir, frames_root):
    video_path = os.path.join(videos_dir, video_name + ".mp4")
    out_dir = os.path.join(frames_root, video_name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in frame_numbers:
            out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_idx += 1
    cap.release()
    print(f"{video_name}: saved {saved}/{len(frame_numbers)} annotated frames")


if __name__ == "__main__":
    needed = get_needed_frames(CSV_PATH)
    for video_name, frame_numbers in needed.items():
        extract_video(video_name, frame_numbers, VIDEOS_DIR, FRAMES_ROOT)