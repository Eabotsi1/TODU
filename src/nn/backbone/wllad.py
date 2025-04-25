
import torch
import torch.nn as nn

class WLLAD(nn.Module):
    """
    Weather-Resilient & Low-Light Adaptive Detection:
    fuses RGB and optional IR streams.
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.rgb_conv = nn.Conv2d(3, d_model, kernel_size=3, padding=1)
        self.ir_conv = nn.Conv2d(1, d_model, kernel_size=3, padding=1)
        self.fusion = nn.Conv2d(2*d_model, d_model, kernel_size=1)

    def forward(self,
                rgb: torch.Tensor,
                ir: torch.Tensor = None) -> torch.Tensor:
        r = self.rgb_conv(rgb)
        if ir is not None:
            i = self.ir_conv(ir)
            return self.fusion(torch.cat([r, i], dim=1))
        return r