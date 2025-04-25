import torch
import torch.nn as nn

class AMSRP(nn.Module):
    """
    Adaptive Multi-Scale Region Processing:
    fuses fine and coarse features using saliency.
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.saliency = nn.Conv2d(d_model, 1, kernel_size=1)
        self.fine = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.coarse = nn.Conv2d(d_model, d_model, kernel_size=5, padding=2)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        S = torch.sigmoid(self.saliency(feat))
        fine_feat = self.fine(feat)
        coarse_feat = self.coarse(feat)
        return S * fine_feat + (1 - S) * coarse_feat