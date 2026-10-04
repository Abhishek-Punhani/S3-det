"""S3Net backbone: stem (stride 4) + three SSB stages producing C2/C3/C4
at strides 4, 8, 16 respectively."""
import torch.nn as nn
from modules import ConvBNAct, SSB


class S3Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        stem_ch = cfg.stem_channels
        c2_ch, c3_ch, c4_ch = cfg.stage_channels

        self.stem = nn.Sequential(
            ConvBNAct(3, stem_ch, k=3, s=2, act=True),
            ConvBNAct(stem_ch, c2_ch, k=3, s=2, act=True),
        )  # -> stride 4, channels = c2_ch

        self.stage2 = self._make_stage(c2_ch, c2_ch, cfg.stage_blocks[0], stride=1, cfg=cfg)
        self.stage3 = self._make_stage(c2_ch, c3_ch, cfg.stage_blocks[1], stride=2, cfg=cfg)
        self.stage4 = self._make_stage(c3_ch, c4_ch, cfg.stage_blocks[2], stride=2, cfg=cfg)

    @staticmethod
    def _make_stage(in_ch, out_ch, n_blocks, stride, cfg):
        layers = [SSB(in_ch, out_ch, stride=stride,
                       expansion=cfg.ssb_expansion, sfg_reduction=cfg.sfg_reduction)]
        for _ in range(n_blocks - 1):
            layers.append(SSB(out_ch, out_ch, stride=1,
                               expansion=cfg.ssb_expansion, sfg_reduction=cfg.sfg_reduction))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.stage2(x)    # stride 4
        c3 = self.stage3(c2)   # stride 8
        c4 = self.stage4(c3)   # stride 16
        return c2, c3, c4


class TimmBackbone(nn.Module):
    """
    Adapter for PyTorch Image Models (timm).
    Extracts P2, P3, P4 feature maps (strides 4, 8, 16) for S3-Det.
    """
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        import timm
        # out_indices=(1, 2, 3) generally corresponds to strides (4, 8, 16) 
        # in standard hierarchical vision models (ResNet, EfficientNet, MobileNet)
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=(1, 2, 3)
        )
        # Store the extracted channel counts so the neck can adapt
        self.channels = self.model.feature_info.channels()

    def forward(self, x):
        features = self.model(x)
        # Return C2 (stride 4), C3 (stride 8), C4 (stride 16)
        return features[0], features[1], features[2]
