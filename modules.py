"""
Building blocks of S3Net: SFG (Spectral/Frequency Gating), GSA (Global
Spatial / coordinate Attention), the SSB (Spectro-Spatial Block) that
combines them in an inverted-residual shell, and the IRU (Implicit
Recursive Unit) gate used (with SHARED weights) inside the IRFA neck.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, groups=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.Hardswish(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SFG(nn.Module):
    """Spectral/Frequency Gating.

    X_lp = AvgPool(X)               (low-frequency component)
    X_hp = X - X_lp                 (high-frequency component)
    alpha = Sigmoid(W2 * ReLU(W1 * GAP(X)))   (channel attention gate)
    out = X + alpha * X_hp
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        hidden = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        low = self.avgpool(x)
        high = x - low
        a = self.gap(x)
        a = F.relu(self.fc1(a), inplace=True)
        a = torch.sigmoid(self.fc2(a))
        return x + a * high


class GSA(nn.Module):
    """Global (coordinate) Spatial Attention.

    Pools along H and W separately (instead of a single global pool) to
    retain positional information, producing a per-row gate g_h and a
    per-column gate g_w:  out = X * g_h * g_w
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.reduce = ConvBNAct(channels, hidden, k=1, act=True)
        self.gate_h = nn.Conv2d(hidden, channels, 1)
        self.gate_w = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        z_h = x.mean(dim=3, keepdim=True)                     # (B,C,H,1)
        z_w = x.mean(dim=2, keepdim=True).transpose(2, 3)      # (B,C,W,1)
        z = torch.cat([z_h, z_w], dim=2)                       # (B,C,H+W,1)
        z = self.reduce(z)                                     # (B,hidden,H+W,1)
        z_h, z_w = torch.split(z, [h, w], dim=2)
        z_w = z_w.transpose(2, 3)                               # (B,hidden,1,W)
        g_h = torch.sigmoid(self.gate_h(z_h))                   # (B,C,H,1)
        g_w = torch.sigmoid(self.gate_w(z_w))                   # (B,C,1,W)
        return x * g_h * g_w


class SSB(nn.Module):
    """Spectro-Spatial Block: 1x1 expansion -> 3x3 depthwise -> SFG -> GSA
    -> 1x1 projection, with a residual connection when shape-compatible."""
    def __init__(self, in_ch, out_ch, stride=1, expansion=4, sfg_reduction=8):
        super().__init__()
        mid_ch = in_ch * expansion
        self.expand = ConvBNAct(in_ch, mid_ch, k=1, act=True)
        self.dwconv = ConvBNAct(mid_ch, mid_ch, k=3, s=stride, groups=mid_ch, act=True)
        self.sfg = SFG(mid_ch, reduction=sfg_reduction)
        self.gsa = GSA(mid_ch, reduction=sfg_reduction)
        self.project = ConvBNAct(mid_ch, out_ch, k=1, act=False)
        self.use_residual = (stride == 1 and in_ch == out_ch)

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.dwconv(out)
        out = self.sfg(out)
        out = self.gsa(out)
        out = self.project(out)
        if self.use_residual:
            out = out + identity
        return out


class IRU(nn.Module):
    """Implicit Recursive Unit gate.

    psi(F) = Sigmoid( K_pw( BN( ReLU( BN( K_dw(F) ) ) ) ) )

    A single instance of this module is instantiated ONCE and reused
    (shared weights) at every fusion point in the IRFA neck.
    """
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x):
        g = self.dw(x)
        g = self.bn1(g)
        g = F.relu(g, inplace=True)
        g = self.bn2(g)
        g = self.pw(g)
        return torch.sigmoid(g)
