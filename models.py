"""The networks for Task A. Input and output are maps of shape (batch, n, n)."""
import torch
from torch import nn


class MLP(nn.Module):
    """Fully connected: the 32 x 32 map is flattened to 1024 numbers.
    1024 -> 512 -> 512 -> 1024, then reshaped back to 32 x 32."""

    def __init__(self, n=32, width=512):
        super().__init__()
        self.n = n
        self.net = nn.Sequential(
            nn.Linear(n * n, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, n * n))

    def forward(self, x):                          # x: (B, 32, 32)
        return self.net(x.flatten(1)).view(-1, self.n, self.n)


def conv(ci, co):
    """Two 3 x 3 convolution layers; zero padding keeps the grid size."""
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.GELU(),
        nn.Conv2d(co, co, 3, padding=1), nn.GELU())


class UNet(nn.Module):
    """Fine grid -> coarse grids -> fine grid, with two skip connections.
    Only convolutions, pooling and upsampling: any n divisible by 4 works.
    skips=False removes the two skip connections (same everything else)."""

    def __init__(self, skips=True):
        super().__init__()
        self.skips = skips
        k = 1 if skips else 0
        self.e1, self.e2, self.e3 = conv(1, 32), conv(32, 64), conv(64, 128)
        self.d2 = conv(128 + k * 64, 64)
        self.d1 = conv(64 + k * 32, 32)
        self.out = nn.Conv2d(32, 1, 1)             # 1 x 1 conv: 32 channels -> stress
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):                          # x: (B, 32, 32)
        h1 = self.e1(x[:, None])                   # (B, 32, 32, 32)
        h2 = self.e2(self.pool(h1))                # (B, 64, 16, 16)
        h3 = self.e3(self.pool(h2))                # (B, 128, 8, 8)
        g2 = self.up(h3)                           # (B, 128, 16, 16)
        g2 = self.d2(torch.cat([g2, h2], 1) if self.skips else g2)   # (B, 64, 16, 16)
        g1 = self.up(g2)                           # (B, 64, 32, 32)
        g1 = self.d1(torch.cat([g1, h1], 1) if self.skips else g1)   # (B, 32, 32, 32)
        return self.out(g1)[:, 0]                  # (B, 32, 32)


MODELS = {
    "mlp": MLP,
    "unet": lambda: UNet(skips=True),
    "unet_noskip": lambda: UNet(skips=False),
}


def count_weights(model):
    return sum(p.numel() for p in model.parameters())
