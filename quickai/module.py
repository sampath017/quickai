import torch.nn.functional as F
import torch
from torch import nn
from .utils import accuracy


class ToyNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.num_classes = num_classes
        self.param = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        logits = self.param * \
            torch.empty((x.shape[0], self.num_classes), dtype=torch.float)

        return logits


class QuickModule:
    def __init__(self, device=None):
        if device in ["cpu", "cuda"]:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, batch):
        self.model = self.model.to(self.device)
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = accuracy(logits, y)

        return loss, acc

    def training_step(self, batch):
        pass

    def validation_step(self, batch):
        pass
