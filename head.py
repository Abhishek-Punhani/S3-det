"""LCR-Head: lightweight decoupled classification/regression head.
Classification uses 3x3 depthwise convs, regression uses 5x5 depthwise
convs (larger receptive field for box regression, as in the paper).
Head weights are shared across FPN levels for parameter efficiency; a
learnable per-level scale rescales the regression logits (standard
practice in FCOS/GFL-style detectors, since a single set of weights must
serve feature maps of different strides)."""
import math
import torch
import torch.nn as nn
from modules import ConvBNAct


class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x):
        return x * self.scale


class LCRHead(nn.Module):
    def __init__(self, channels, num_classes, reg_max, stacked_convs, num_levels):
        super().__init__()
        self.reg_max = reg_max

        cls_layers = [ConvBNAct(channels, channels, k=3, groups=channels, act=True)
                      for _ in range(stacked_convs)]
        reg_layers = [ConvBNAct(channels, channels, k=5, groups=channels, act=True)
                      for _ in range(stacked_convs)]
        self.cls_convs = nn.Sequential(*cls_layers)
        self.reg_convs = nn.Sequential(*reg_layers)

        self.cls_pred = nn.Conv2d(channels, num_classes, 3, padding=1)
        self.reg_pred = nn.Conv2d(channels, 4 * (reg_max + 1), 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(num_levels)])

        # standard focal-loss bias init so training doesn't start with a huge
        # number of confident false positives
        prior_prob = 0.01
        bias_init = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_init)

    def forward(self, feats):
        cls_outs, reg_outs = [], []
        for i, f in enumerate(feats):
            cls_feat = self.cls_convs(f)
            reg_feat = self.reg_convs(f)
            cls_outs.append(self.cls_pred(cls_feat))
            reg_outs.append(self.scales[i](self.reg_pred(reg_feat)))
        return cls_outs, reg_outs
