"""
PointNet++ (SSG) part-segmentation backbone, adapted as a Pointcept backbone.

Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
(models/pointnet2_part_seg_ssg.py).

Interface differences bridged here:
  * Pointcept feeds a `Point` with a flat, batch-concatenated layout
    (feat/coord shape (N_total, C), boundaries given by `offset`).
    PointNet++ wants a dense, fixed-size layout (B, C, N). We pad each sample
    up to the batch max by cyclically repeating its own points (padding points
    are therefore valid coordinates and do not corrupt FPS / ball-query), run
    the network, then unpad back to the original flat order.
  * The SSG part-seg head requires a 16-d ShapeNet category one-hot. Per the
    chosen configuration we feed a fixed zero vector (single-category training),
    keeping the original channel arithmetic unchanged.
  * The final classifier (dropout + Conv1d -> log_softmax) is dropped. We expose
    the 128-d per-point feature so the `DefaultKeypointer*` wrapper's own head
    (Linear(128 -> num_keypoints)) projects it. Set `backbone_out_channels=128`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.builder import MODELS
from pointcept.models.utils.structure import Point
from .pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


@MODELS.register_module("PointNet2-PartSeg")
class PointNet2PartSeg(nn.Module):
    def __init__(self, in_channels=3, num_category=16):
        super().__init__()
        assert in_channels == 3, "PointNet2-PartSeg adapter expects xyz-only input"
        self.in_channels = in_channels
        self.num_category = num_category

        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=6, mlp=[64, 64, 128], group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True,
        )
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(
            in_channel=128 + num_category + 6, mlp=[128, 128, 128]
        )
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)

    @staticmethod
    def _to_dense(coord, offset):
        """(N_total, 3) + offset -> dense (B, N_max, 3) and per-sample counts.

        Short samples are padded by cyclically repeating their own points.
        """
        offset_pad = F.pad(offset, (1, 0))
        counts = offset_pad[1:] - offset_pad[:-1]
        B = counts.numel()
        n_max = int(counts.max().item())
        dense = coord.new_zeros(B, n_max, coord.shape[1])
        ar = torch.arange(n_max, device=coord.device)
        for i in range(B):
            s, n = int(offset_pad[i].item()), int(counts[i].item())
            idx = ar % n  # cyclic repeat to fill padding
            dense[i] = coord[s:s + n][idx]
        return dense, counts

    @staticmethod
    def _to_flat(dense_feat, counts):
        """dense (B, N_max, C) -> flat (N_total, C), dropping padding, orig order."""
        return torch.cat(
            [dense_feat[i, : int(counts[i].item())] for i in range(counts.numel())],
            dim=0,
        )

    def forward(self, point):
        coord = point["coord"]
        offset = point["offset"]

        xyz, counts = self._to_dense(coord, offset)  # (B, N, 3)
        xyz = xyz.permute(0, 2, 1).contiguous()      # (B, 3, N)
        B, _, N = xyz.shape

        l0_xyz = xyz
        l0_points = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        # Fixed zero category one-hot (single-category training).
        cls_one_hot = xyz.new_zeros(B, self.num_category, N)
        l0_points = self.fp1(
            l0_xyz, l1_xyz,
            torch.cat([cls_one_hot, l0_xyz, l0_points], dim=1),
            l1_points,
        )
        feat = F.relu(self.bn1(self.conv1(l0_points)))  # (B, 128, N)

        feat = feat.permute(0, 2, 1).contiguous()       # (B, N, 128)
        point["feat"] = self._to_flat(feat, counts)     # (N_total, 128)
        return point
