import torch
import torch.nn as nn
import torch.nn.functional as F


class Boryviter(nn.Module):

    def __init__(self, num_anchors=2, num_classes=1):
        super().__init__()

        out_channels = num_anchors * (5 + num_classes) # 12 if num_anchors didn't change

        self.features = nn.Sequential( 
            # 0: conv 16, 3x3, 2
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),

            # 1: maxpool 2x2, stride 2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 2: conv 32, 3x3, 1
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),

            # 3: maxpool 2x2, stride 2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 4: conv 32, 1x1, 1
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),

            # 5: maxpool 2x2, stride 2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 6: conv 64, 3x3, 1
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            # 7: conv 64, 1x1, 1
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            # 8: maxpool 2x2, stride 2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 9: conv 128, 1x1, 1
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),

            # 10: conv 128, 1x1, 1
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # 11: conv 12, 1x1, 1
        self.head = nn.Conv2d(128, out_channels, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        x = self.features(x)
        x = self.head(x) #meow
        return x 

if __name__ == "__main__":
    model = Boryviter()
    dummy = torch.randn(1, 3, 320, 416) # fake image for test
    out = model(dummy)
    print("Output shape:", out.shape) # 1, 12, 10, 13
    n_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", n_params) # 56108
