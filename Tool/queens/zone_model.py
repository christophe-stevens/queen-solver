import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from queens.config import SIZES, MAX_ZONES


class ZoneModel(nn.Module):
    """Puzzle image -> grid size, and zone number for each cell (any size in SIZES)."""

    def __init__(self, size=280):
        super().__init__()
        self.size = size  # images are resized to size x size pixels (20 px per cell at 14x14)
        # Zone branch: pixels -> features, pooled to one vector per cell
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.head = nn.Conv2d(32, MAX_ZONES, 1)  # zone scores per cell
        # Size branch: whole image -> one score per possible grid size
        self.size_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),  # one summary vector per image
            nn.Linear(32, len(SIZES)))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, m):
        # (B,3,size,size) images, grid size m -> (B,MAX_ZONES,m,m) zone scores
        return self.head(F.adaptive_avg_pool2d(self.features(x), m))

    def size_scores(self, x):
        # (B,3,size,size) images -> (B,len(SIZES)) scores, one per grid size
        return self.size_net(x)

    def detect_size(self, x):
        # (B,3,size,size) images -> list of detected grid sizes
        return [SIZES[i] for i in self.size_scores(x).argmax(1).tolist()]

    def image_to_tensor(self, path):
        # PNG file -> (3,size,size) tensor scaled to 0-1
        img = Image.open(path).convert("RGB").resize((self.size, self.size))
        return torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255

    def read_image(self, path, m=None):
        # PNG file -> (m,m) zone numbers; detects m if not given
        x = self.image_to_tensor(path).unsqueeze(0).to(self.device)
        with torch.no_grad():
            m = m or self.detect_size_by_fit(x)[0]
            return self(x, m).argmax(1)[0]

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
        return self

    def fit_scores(self, x):
        # (B,3,size,size) -> (B,len(SIZES)): average certainty of the zone reading at each size
        return torch.stack([F.softmax(self(x, m), dim=1).max(1).values.mean((1, 2))
                            for m in SIZES], dim=1)

    def detect_size_by_fit(self, x, tol=0.005):
        # Smallest size whose zone reading is (almost) as confident as the best one
        conf = self.fit_scores(x)
        ok = conf >= conf.max(1, keepdim=True).values - tol
        return [SIZES[row.nonzero()[0].item()] for row in ok]