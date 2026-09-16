"""
Processing data for Boryviter
Turning different data sources into one common format.

What the script does:

1. Convert box format:
    Boryviter detector's anchors regress center + size (x_center, y_center, w, h)
    w = x2 - x1
    h = y2 - y1
    x_center = x1 + w/2
    y_center = y1 + h/2

2. Collapse to one class: drop the species label, every box is "bird"

3. Normalize to fractions of the frame: 
    image width = x_center/width
    image_height = y_center/height
    Boxes are still valid after resizing frame to the network's input

4. Assign each box to its grid cell + best-matching anchor

"""

import csv
import ast
from collections import defaultdict
import random
from pathlib import Path

import cv2
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

# CNN design - x input, 5 downsamples
IMG_W, IMG_H = 416, 320
GRID_W, GRID_H = 13, 10

# Anchors are fractions of the image, replace with k-means later
ANCHORS = [
    (0.2, 0.125), # spread wings
    (0.1, 0.14)   # closed wings
]
NUM_ANCHORS = len(ANCHORS)


CSV_PATH = "wetland/bounding_boxes.csv"
VIDEOS_DIR = "wetland/videos"
FRAMES_ROOT = "wetland/frames"

"""
Wetland:
species_id;species;video_name;frame;bounding_boxes
0;White Wagtail;001-white_wagtail;0;[(364.62, 191.11, 574.08, 394.75, 1, 0)]
                                    [(X_max,Y_max,X_min,Y_min,Behavior_id,Bird_id)]
"""

def box_iou(w1, h1, w2, h2):
    # IoU of two boxes centered on the same point, shape comparison to decide which anchor a ground-truth box belongs to
    inter = min(w1, w2) * min(h1, h2)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0

# Normalized 0-1: instead of pixels, express it as what fraction of the frame this is

def best_anchor(bw, bh):
    ious = [box_iou(bw, bh, aw, ah) for aw, ah in ANCHORS]
    return max(range(NUM_ANCHORS), key=lambda i: ious[i])

def make_target(boxes):
    target = torch.zeros(NUM_ANCHORS * 5, GRID_H, GRID_W)
    for cx, cy, bw, bh in boxes:
        gx = min(int(cx * GRID_W), GRID_W - 1)
        gy = min(int(cy * GRID_H), GRID_H - 1)
        a = best_anchor(bw, bh)
        base = a * 5

        if target[base, gy, gx] == 1:
            continue # this cell+anchor slot already taken this frame

        tx = cx * GRID_W - gx
        ty = cy * GRID_H - gy
        target[base, gy, gx] = 1
        target[base + 1: base + 5, gy, gx] = torch.tensor([tx, ty, bw, bh])

    return target

def split_videos(csv_path: str, val_ratio: float = 0.2, seed: int = 42):
    video_names = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f, delimiter=";"):
            video_names.add(row["video_name"])
 
    video_names = sorted(video_names)
    random.Random(seed).shuffle(video_names)
 
    n_val = max(1, int(len(video_names) * val_ratio))
    return set(video_names[n_val:]), set(video_names[:n_val])


class WetlandBirdsDataset(Dataset):
    """WetlandBirds CSV + pre-extracted frames -> (image, target) pairs.
 
    Expects pre-extracted frames: frames_root/<video_name>/frame_000001.jpg
    """
 
    def __init__(self, frames_root: str, csv_path: str, video_filter=None, augment: bool = False):
        self.frames_root = Path(frames_root)
        self.augment = augment
 
        self.annotations = defaultdict(list)
        with open(csv_path) as f:
            for row in csv.DictReader(f, delimiter=";"):
                video_name = row["video_name"]
                if video_filter is not None and video_name not in video_filter:
                    continue
                key = (video_name, int(row["frame"]))
                boxes = ast.literal_eval(row["bounding_boxes"])
                for x_min, y_min, x_max, y_max, behavior_id, bird_id in boxes:
                    self.annotations[key].append(
                        {"x1": x_min, "y1": y_min, "x2": x_max, "y2": y_max}
                    )
        self.keys = sorted(self.annotations.keys())
 
    def __len__(self):
        return len(self.keys)
 
    def __getitem__(self, idx):
        video_name, frame_id = self.keys[idx]
        img_path = self.frames_root / video_name / f"frame_{frame_id:06d}.jpg"
 
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"failed to read image: {img_path}")
        h, w = image.shape[:2]
        boxes = self.annotations[video_name, frame_id]
 
        do_flip = self.augment and random.random() < 0.5
        if do_flip:
            image = cv2.flip(image, 1)
 
        image = cv2.resize(image, (IMG_W, IMG_H)) 
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
 
        norm_boxes = []
        for box in boxes:
            x1, x2 = box["x1"], box["x2"]
            if do_flip:
                x1, x2 = w - box["x2"], w - box["x1"]
            cx = (x1 + x2) / 2 / w
            cy = (box["y1"] + box["y2"]) / 2 / h
            bw = (x2 - x1) / w
            bh = (box["y2"] - box["y1"]) / h
            norm_boxes.append((cx, cy, bw, bh))
 
        target = make_target(norm_boxes)
        return image_t, target
 
 
class OwnFootageDataset(Dataset):
    """Your own hand-labeled footage: images/*.jpg + labels/*.txt
    (one .txt per image, lines of 'class cx cy w h', already normalized --
    the format convert_wetlandbirds.py writes)."""
 
    def __init__(self, images_dir: str, labels_dir: str, augment: bool = False):
        self.labels_dir = Path(labels_dir)
        self.augment = augment
        self.image_files = sorted(Path(images_dir).glob("*.jpg"))
 
    def __len__(self):
        return len(self.image_files)
 
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.labels_dir / (img_path.stem + ".txt")
 
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"failed to read image: {img_path}")
 
        boxes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    _, cx, cy, bw, bh = map(float, line.split())
                    boxes.append((cx, cy, bw, bh))
 
        do_flip = self.augment and random.random() < 0.5
        if do_flip:
            image = cv2.flip(image, 1)
            boxes = [(1 - cx, cy, bw, bh) for cx, cy, bw, bh in boxes]
 
        image = cv2.resize(image, (IMG_W, IMG_H))
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
 
        target = make_target(boxes)
        return image_t, target

def get_dataloaders(wetlandbirds_frames, wetlandbirds_csv, own_images=None, own_labels=None, batch_size=16, val_ratio=0.2):
    train_videos, val_videos = split_videos(wetlandbirds_csv, val_ratio=val_ratio)
    train_sets = [WetlandBirdsDataset(wetlandbirds_frames, wetlandbirds_csv, video_filter=train_videos, augment=True)]
    val_sets = [WetlandBirdsDataset(wetlandbirds_frames, wetlandbirds_csv, video_filter=val_videos, augment=False)]

    if own_images is not None:
        train_sets.append(OwnFootageDataset(own_images, own_labels, augment=True))

    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ConcatDataset(val_sets), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader