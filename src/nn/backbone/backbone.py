
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

class HybridCNNTransformerBackbone(nn.Module):
    """
    Combines EfficientNet-B0 features with Transformer encoder for global context.
    """
    def __init__(self,
                 pretrained: bool = False,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4):
        super().__init__()
        self.cnn = efficientnet_b0(pretrained=pretrained).features
        self.proj = nn.Conv2d(1280, d_model, kernel_size=1)
        self.pos_enc = nn.Parameter(torch.randn(1, d_model, 1, 1))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_channels = d_model  # or whatever the actual output channels of your model are


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)  # [B,1280,H/32,W/32]
        B, C, H, W = feat.shape
        proj = self.proj(feat) + self.pos_enc
        seq = proj.flatten(2).permute(2, 0, 1)  # [H*W, B, C]
        trans = self.transformer(seq)
        out = trans.permute(1, 2, 0).view(B, -1, H, W)
        return out