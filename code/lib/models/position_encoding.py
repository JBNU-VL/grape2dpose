import math
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    Source: https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
    """

    def __init__(self, d_model=64, temperature=10000, normalize=True, scale=2 * math.pi):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, x):
        _, _, h, w = x.shape
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = self.d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        t_ = torch.div(dim_t, 2, rounding_mode='floor')
        dim_t = self.temperature ** (2 * (t_) / one_direction_feats)
        #         dim_t = temperature ** (2 * torch.div((dim_t // 2),one_direction_feats, rounding_mode='floor'))

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # pos = pos.flatten(2).permute(2, 0, 1)
        pos = pos.flatten(2)

        return nn.Parameter(pos, requires_grad=False).to(x.device).type(torch.cuda.HalfTensor)