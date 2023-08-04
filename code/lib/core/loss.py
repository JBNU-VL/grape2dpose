from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

logger = logging.getLogger(__name__)


class HeatmapLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        loss = ((pred - gt) ** 2) * mask
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)
        return loss

class CenterLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        loss = ((pred - gt) ** 2) * mask
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)
        return loss

class OffsetsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def smooth_l1_loss(self, pred, gt, beta=1. / 9):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < beta
        loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
        return loss

    def forward(self, pred, gt, weights):
        assert pred.size() == gt.size()
        num_pos = torch.nonzero(weights > 0).size()[0]
        loss = self.smooth_l1_loss(pred, gt) * weights
        if num_pos == 0:
            num_pos = 1.
        loss = loss.sum() / num_pos
        return loss


class FocalSigmoidLossFunc(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''

    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, label, alpha, gamma):
        probs = torch.sigmoid(logits)
        coeff = (label - probs).abs_().pow_(gamma).neg_()
        log_probs = torch.where(logits >= 0,
                                F.softplus(logits, -1, 50),
                                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                                  -logits + F.softplus(logits, -1, 50),
                                  -F.softplus(logits, 1, 50))
        ce_term1 = log_probs.mul_(label).mul_(alpha)
        ce_term2 = log_1_probs.mul_(1. - label).mul_(1. - alpha)
        ce = ce_term1.add_(ce_term2)
        loss = ce * coeff

        ctx.vars = (coeff, probs, ce, label, gamma, alpha)

        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient of focal loss
        '''
        (coeff, probs, ce, label, gamma, alpha) = ctx.vars

        d_coeff = (label - probs).abs_().pow_(gamma - 1.).mul_(gamma)
        d_coeff.mul_(probs).mul_(1. - probs)
        d_coeff = torch.where(label < probs, d_coeff.neg(), d_coeff)
        term1 = d_coeff.mul_(ce)

        d_ce = label * alpha
        d_ce.sub_(probs.mul_((label * alpha).mul_(2).add_(1).sub_(label).sub_(alpha)))
        term2 = d_ce.mul(coeff)

        grads = term1.add_(term2)
        grads.mul_(grad_output)

        return grads, None, None, None


class FocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV2()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        loss = FocalSigmoidLossFunc.apply(logits, label, self.alpha, self.gamma)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


class FeatureLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1):
        super(FeatureLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def hierarchical_pool(self, heatmap):
        pool1 = torch.nn.MaxPool2d(3, 1, 1)
        pool2 = torch.nn.MaxPool2d(5, 1, 2)
        pool3 = torch.nn.MaxPool2d(7, 1, 3)
        map_size = (heatmap.shape[1] + heatmap.shape[2]) / 2.0
        if map_size > 300:
            maxm = pool3(heatmap[None, :, :, :])
        elif map_size > 200:
            maxm = pool2(heatmap[None, :, :, :])
        else:
            maxm = pool1(heatmap[None, :, :, :])
        return maxm

    def get_maximum_from_heatmap(self, heatmap):
        maxm = self.hierarchical_pool(heatmap)
        maxm = torch.eq(maxm, heatmap).float()
        heatmap = heatmap * maxm
        hh = heatmap[0][0]
        nonzero_idx = torch.nonzero(hh)
        coordinates = nonzero_idx.tolist()
        return coordinates

    def forward(self, feature_map, center_heatmap):
        batch_size, num_channels, height, width = feature_map.size()
        assert center_heatmap.size() == (batch_size, 1, height, width), "Feature map size and heatmap size do not match"

        dist_same_class = 0.0
        dist_between_class = 0.0
        for i in range(batch_size):
            features = feature_map[i]
            coordinates = self.get_maximum_from_heatmap(center_heatmap[i])

            region_size = 6
            N = len(coordinates)
            if N == 0:
                continue

            half_size = region_size // 2

            # Padding
            pad_left = pad_top = half_size
            pad_right = pad_bottom = half_size
            features = torch.nn.functional.pad(features, [pad_left, pad_right, pad_top, pad_bottom])

            regions = []
            for i in range(N):
                x = coordinates[i][0]
                y = coordinates[i][1]
                x = x + half_size
                y = y + half_size
                region = features[:, y - half_size:y + half_size + 1, x - half_size:x + half_size + 1]
                regions.append(region)

            regions = torch.stack(regions)
            loss_in = 0.0
            for i in range (regions.shape[0]):
                mean_feature = torch.mean(regions[i], dim=0, keepdim=True)
                dist = torch.norm(regions[i] - mean_feature, dim=(1, 2))
                loss_in += torch.mean(dist)
            loss_in /= regions.shape[0]
            dist_same_class += loss_in

            loss_out = 0.0
            for i in range (regions.shape[0]):
                for j in range(i + 1, regions.shape[0]):
                    mean_feature_i = torch.mean(regions[i], dim=0, keepdim=True)
                    mean_feature_j = torch.mean(regions[j], dim=0, keepdim=True)
                    dist = torch.norm(mean_feature_i - mean_feature_j, dim=(1, 2))
                    loss_out += torch.max(torch.tensor(0.0), 1 - dist)
            if regions.shape[0] * (regions.shape[0] - 1) / 2 == 0:
                loss_out = 0
            else:
                loss_out /= regions.shape[0] * (regions.shape[0] - 1) / 2
            dist_between_class += loss_out
        dist_same_class /= batch_size
        dist_between_class /= batch_size
        loss = self.alpha * dist_same_class + self.beta * dist_between_class

        return loss



class MultiLossFactory(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.bg_weight = cfg.DATASET.BG_WEIGHT

        self.heatmap_loss = HeatmapLoss() if cfg.LOSS.WITH_HEATMAPS_LOSS else None
        self.heatmap_loss_factor = cfg.LOSS.HEATMAPS_LOSS_FACTOR

        self.center_loss = CenterLoss()

        self.offset_loss = OffsetsLoss() if cfg.LOSS.WITH_OFFSETS_LOSS else None
        self.offset_loss_factor = cfg.LOSS.OFFSETS_LOSS_FACTOR

        self.visibility_loss = FocalLoss() if cfg.LOSS.WITH_VISIBILITY_LOSS else None
        self.visibility_loss_factor = cfg.LOSS.VISIBILITY_LOSS_FACTOR

        self.feature_loss = FeatureLoss() if cfg.LOSS.WITH_FEATURE_LOSS else None
        self.feature_loss_factor = cfg.LOSS.FEATURE_LOSS_FACTOR

    def forward(self, pheatmap_init, poffset_init, \
                poffset_final, pvisibility, heatmap, center_mask, mask, offset, offset_w, visibility_map, features, center_heatmap):
        if self.heatmap_loss:
            heatmap_loss_init = self.heatmap_loss(pheatmap_init, heatmap, mask)
            heatmap_loss_init = heatmap_loss_init * 10

        else:
            heatmap_loss_init = None

        if self.offset_loss:
            offset_loss_init = self.offset_loss(poffset_init, offset, offset_w)
            offset_loss_init = offset_loss_init * 0.3
            offset_loss_final = self.offset_loss(poffset_final, offset, offset_w)
            offset_loss_final = offset_loss_final * 0.3
        else:
            offset_loss_init = None
            offset_loss_final = None


        if self.visibility_loss:
            visibility_loss = self.visibility_loss(pvisibility, visibility_map)
            visibility_loss = visibility_loss * self.visibility_loss_factor
        else:
            visibility_loss = None

        if self.feature_loss:
            # feature_loss = self.feature_loss(features, center_heatmap)
            # feature_loss = feature_loss * self.feature_loss_factor
            feature_loss = None
        else:
            feature_loss = None


        return heatmap_loss_init, \
               offset_loss_init, offset_loss_final, visibility_loss, feature_loss