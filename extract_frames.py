import cv2
from pathlib import Path

VIDEOS_DIR = Path("dataset/videos")
FRAMES_ROOT = Path("dataset/frames")

for video_path in VIDEOS_DIR.glob("*.mp4"):
    video_name = video_path.stem
    out_dir = FRAMES_ROOT / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"frame_{frame_id:06d}.jpg"), frame)
        frame_id += 1
    cap.release()
    print(f"{video_name}: {frame_id} frames")
