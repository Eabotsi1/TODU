
import torch
import torch.nn as nn
from .backbone import HybridCNNTransformerBackbone
from .amsrp import AMSRP
from .wllad import WLLAD
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

class Detector(nn.Module):
    """
    Builds the detection model by integrating backbone, AMSRP, WL-LAD into Faster R-CNN.
    """
    def __init__(self, num_classes: int, pretrained_backbone: bool = False):
        super().__init__()
        self.backbone = HybridCNNTransformerBackbone(pretrained=pretrained_backbone)
        # Feature extractor returns 256 channels
        backbone_out = 256
        anchor_gen = AnchorGenerator(sizes=((32, 64, 128),), aspect_ratios=((0.5, 1.0, 2.0),))
        self.faster_rcnn = FasterRCNN(
            backbone=self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_gen
        )
        self.amsrp = AMSRP(d_model=backbone_out)
        self.wllad = WLLAD(d_model=backbone_out)

    def forward(self, images, targets=None):
        # Apply WL-LAD if IR channel passed
        processed = []
        for img in images:
            if img.shape[0] == 4:
                rgb, ir = img[:3], img[3:].unsqueeze(0)
                feat = self.wllad(rgb.unsqueeze(0), ir)
            else:
                feat = self.wllad(img.unsqueeze(0))
            processed.append(feat.squeeze(0))

        # Pass through backbone and AMSRP inside FasterRCNN
        return self.faster_rcnn(processed, targets)