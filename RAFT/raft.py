# RAFT: Recurrent All-Pairs Field Transforms for Optical Flow
# https://zhuanlan.zhihu.com/p/222012743


# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock
from .extractor import BasicEncoder
from .corr import CorrBlock
from .utils.utils import coords_grid
import pdb


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        # pdb.set_trace()

        # pp args.small -- False
        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if "dropout" not in args._get_kwargs():  # True
            args.dropout = 0

        if "alternate_corr" not in args._get_kwargs():  # True
            args.alternate_corr = False

        # hdim,cdim -- (128, 128)
        self.fnet = BasicEncoder(output_dim=256, norm_fn="instance", dropout=args.dropout)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn="batch", dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=20):
        """Estimate optical flow between pair of frames"""

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        fmap1, fmap2 = self.fnet([image1, image2])

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        # flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            coords1 = coords1 + delta_flow

            flow_up = self.upsample_flow(coords1 - coords0, up_mask)

        return flow_up
