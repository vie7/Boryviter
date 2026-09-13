import torch
from torch.utils.data import DataLoader

from model import CNN, detection_loss
from dataset import WetlandBirdsDataset, split_videos

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50

model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

train_videos, val_videos = split_videos("wetland/bounding_boxes.csv", val_ratio=0.2)

train_dataset = WetlandBirdsDataset("wetland/frames", "wetland/bounding_boxes.csv", video_filter=train_videos, augment=True)
val_dataset   = WetlandBirdsDataset("wetland/frames", "wetland/bounding_boxes.csv", video_filter=val_videos)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)


for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        preds = model(images)
        loss, parts = detection_loss(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss_total, val_parts_total = 0.0, {"obj": 0, "noobj": 0, "box": 0}
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images)
            vloss, vparts = detection_loss(preds, targets)
            val_loss_total += vloss.item()
            for k in val_parts_total:
                val_parts_total[k] += vparts[k]

    n = len(val_loader)
    print(f"epoch {epoch}: train={loss.item():.4f} val={val_loss_total/n:.4f}")