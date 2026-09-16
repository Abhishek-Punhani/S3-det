"""IRFA neck: lateral 1x1 convs unify C2/C3/C4 to a common width, then a
top-down pass followed by a bottom-up pass fuses adjacent levels. The same
IRU instance (shared weights) gates all four fusion points."""
import torch.nn.functional as F
import torch.nn as nn
from modules import ConvBNAct, IRU


class IRFA(nn.Module):
    def __init__(self, in_channels, neck_channels, cfg):
        super().__init__()
        c2_ch, c3_ch, c4_ch = in_channels
        self.lateral2 = ConvBNAct(c2_ch, neck_channels, k=1, act=False)
        self.lateral3 = ConvBNAct(c3_ch, neck_channels, k=1, act=False)
        self.lateral4 = ConvBNAct(c4_ch, neck_channels, k=1, act=False)

        self.iru = IRU(neck_channels)  # single shared instance, applied 4x

        self.down2 = ConvBNAct(neck_channels, neck_channels, k=3, s=2,
                                groups=neck_channels, act=True)
        self.down3 = ConvBNAct(neck_channels, neck_channels, k=3, s=2,
                                groups=neck_channels, act=True)

    def forward(self, c2, c3, c4):
        l2 = self.lateral2(c2)
        l3 = self.lateral3(c3)
        l4 = self.lateral4(c4)

        # ---- top-down: P_i_td = P_i + Upsample(P_{i+1}) * IRU(P_i) (Eq. 13) ----
        p4_td = l4  # coarsest level: no P_5 to fuse from, unchanged
        g3_td = self.iru(l3)                                             # IRU application 1
        up4 = F.interpolate(p4_td, size=l3.shape[-2:], mode="nearest")
        p3_td = l3 + up4 * g3_td

        g2_td = self.iru(l2)                                             # IRU application 2
        up3 = F.interpolate(p3_td, size=l2.shape[-2:], mode="nearest")
        p2_td = l2 + up3 * g2_td

        # ---- bottom-up: N_i_out = N_i + Downsample(N_{i-1}) * IRU(N_i) (Eq. 14) ----
        n2 = p2_td  # finest level: no N_1 to fuse from, unchanged
        g3_bu = self.iru(p3_td)                                          # IRU application 3
        n3 = p3_td + self.down2(n2) * g3_bu

        g4_bu = self.iru(p4_td)                                          # IRU application 4
        n4 = p4_td + self.down3(n3) * g4_bu

        return n2, n3, n4
