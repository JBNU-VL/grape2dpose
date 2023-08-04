from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np


def get_heat_value(pose_coord, heatmap, visibility=True):
    _, h, w = heatmap.shape

    if (visibility == True):
        heatmap_nocenter = heatmap.flatten(1, 2).transpose(0, 1)
    else:
        heatmap_nocenter = heatmap[:-1].flatten(1, 2).transpose(0, 1)

    y_b = torch.clamp(torch.floor(pose_coord[:, :, 1]), 0, h - 1).long()
    x_l = torch.clamp(torch.floor(pose_coord[:, :, 0]), 0, w - 1).long()

    heatval = torch.gather(heatmap_nocenter, 0, y_b * w + x_l).unsqueeze(-1)
    return heatval


def cal_area_2_torch(v):
    w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
    h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
    return w * w + h * h


def nms_core(cfg, pose_coord, heat_score):
    num_people, num_joints, _ = pose_coord.shape
    pose_area = cal_area_2_torch(pose_coord)[:, None].repeat(1, num_people * num_joints)
    pose_area = pose_area.reshape(num_people, num_people, num_joints)

    pose_diff = pose_coord[:, None, :, :] - pose_coord
    pose_diff.pow_(2)
    pose_dist = pose_diff.sum(3)
    pose_dist.sqrt_()
    pose_thre = cfg.TEST.NMS_THRE * torch.sqrt(pose_area)
    pose_dist = (pose_dist < pose_thre).sum(2)
    nms_pose = pose_dist > cfg.TEST.NMS_NUM_THRE

    ignored_pose_inds = []
    keep_pose_inds = []
    for i in range(nms_pose.shape[0]):
        if i in ignored_pose_inds:
            continue
        keep_inds = nms_pose[i].nonzero().cpu().numpy()
        keep_inds = [list(kind)[0] for kind in keep_inds]
        keep_scores = heat_score[keep_inds]
        ind = torch.argmax(keep_scores)
        keep_ind = keep_inds[ind]
        if keep_ind in ignored_pose_inds:
            continue
        keep_pose_inds += [keep_ind]
        ignored_pose_inds += list(set(keep_inds) - set(ignored_pose_inds))

    return keep_pose_inds


def delete_row_tensor(tensor, index_del):
    n = tensor.cpu().detach().numpy()
    n = np.delete(n, index_del, 0)
    n = torch.from_numpy(n).cuda()
    return n


def pose_nms(cfg, heatmap_avg, poses, visibilitymap_avg, scales):
    """
    NMS for the regressed poses results.

    Args:
        heatmap_avg (Tensor): Avg of the heatmaps at all scales (1, 1+num_joints, w, h)
        poses (List): Gather of the pose proposals [(num_people, num_joints, 3)]
    """
    scale1_index = sorted(scales, reverse=True).index(1.0)
    pose_norm = poses[scale1_index]
    max_score = pose_norm[:, :, 2].max() if pose_norm.shape[0] else 1

    for i, pose in enumerate(poses):
        if i != scale1_index:
            max_score_scale = pose[:, :, 2].max() if pose.shape[0] else 1
            pose[:, :, 2] = pose[:, :, 2] / max_score_scale * max_score * cfg.TEST.DECREASE

    # remove Nan values
    for i in range(len(poses)):
        remove = []
        for row in range(poses[i].shape[0]):
            if (torch.isnan(poses[i][row][0][0])):
                remove.append(row)
        poses[i] = delete_row_tensor(poses[i], remove)

    pose_score = torch.cat([pose[:, :, 2:] for pose in poses], dim=0)
    pose_coord = torch.cat([pose[:, :, :2] for pose in poses], dim=0)

    if pose_coord.shape[0] == 0:
        return [], []

    num_people, num_joints, _ = pose_coord.shape

    heatval = get_heat_value(pose_coord, heatmap_avg[0], visibility=False)
    heat_score = (torch.sum(heatval, dim=1) / num_joints)[:, 0]

    visibilityval = get_heat_value(pose_coord, visibilitymap_avg[0], visibility=True)

    if (torch.isnan(visibilitymap_avg[0])[0][0][0]):
        for i in range(pose_score.shape[0]):
            pose_score[i] = pose_score[i] * heatval[i]
    else:
        for i in range(pose_score.shape[0]):
            if (pose_score[i][0].item() < 0.4):
                pose_score[i] = pose_score[i] * heatval[i]
            else:
                count = 0
                sum_vis = 0
                for j in visibilityval[i]:
                    if (j >= 0.1):
                        count += 1
                        sum_vis += j
                if (count > 2):
                    score = float(sum_vis.item() / count)
                    pose_score[i] = torch.full((cfg.DATASET.NUM_JOINTS, 1), score)
                else:
                    pose_score[i] = pose_score[i] * heatval[i]

    poses = torch.cat([pose_coord.cpu(), pose_score.cpu(), visibilityval.cpu()], dim=2)

    keep_pose_inds = nms_core(cfg, pose_coord, heat_score)
    poses = poses[keep_pose_inds]
    heat_score = heat_score[keep_pose_inds]

    if len(keep_pose_inds) > cfg.DATASET.MAX_NUM_PEOPLE:
        heat_score, topk_inds = torch.topk(heat_score, cfg.DATASET.MAX_NUM_PEOPLE)
        poses = poses[topk_inds]

    poses = [poses.numpy()]
    scores = [i[:, 2].mean() for i in poses[0]]

    return poses, scores


def pose_nms_new(cfg, heatmap_avg, poses, visibilitymap_avg, scales):
    """
    NMS for the regressed poses results.

    Args:
        heatmap_avg (Tensor): Avg of the heatmaps at all scales (1, 1+num_joints, w, h)
        poses (List): Gather of the pose proposals [(num_people, num_joints, 3)]
    """
    scale1_index = sorted(scales, reverse=True).index(1.0)
    pose_norm = poses[scale1_index]
    max_score = pose_norm[:, :, 2].max() if pose_norm.shape[0] else 1

    for i, pose in enumerate(poses):
        if i != scale1_index:
            max_score_scale = pose[:, :, 2].max() if pose.shape[0] else 1
            pose[:, :, 2] = pose[:, :, 2] / max_score_scale * max_score * cfg.TEST.DECREASE

    # remove Nan values
    for i in range(len(poses)):
        remove = []
        for row in range(poses[i].shape[0]):
            if (torch.isnan(poses[i][row][0][0])):
                remove.append(row)
        poses[i] = delete_row_tensor(poses[i], remove)

    pose_score = torch.cat([pose[:, :, 2:] for pose in poses], dim=0)
    pose_coord = torch.cat([pose[:, :, :2] for pose in poses], dim=0)

    visibilitymap_avg[0] = torch.nan_to_num(visibilitymap_avg[0], nan=0.009)

    if pose_coord.shape[0] == 0:
        return [], []

    num_people, num_joints, _ = pose_coord.shape

    heatval = get_heat_value(pose_coord, heatmap_avg[0], visibility=False)
    heat_score = (torch.sum(heatval, dim=1) / num_joints)[:, 0]

    visibilityval = get_heat_value(pose_coord, visibilitymap_avg[0], visibility=True)

    for i in range(pose_score.shape[0]):
        if (pose_score[i][0].item() < 0.4):
            pose_score[i] = pose_score[i] * heatval[i]
        else:
            count = 0
            sum_vis = 0
            for j in visibilityval[i]:
                if (j >= 0.1):
                    count += 1
                    sum_vis += j
            if (count > 2):
                score = float(sum_vis.item() / count)
                pose_score[i] = torch.full((cfg.DATASET.NUM_JOINTS, 1), score)
            else:
                pose_score[i] = pose_score[i] * heatval[i]

    poses = torch.cat([pose_coord.cpu(), pose_score.cpu(), visibilityval.cpu()], dim=2)

    keep_pose_inds = nms_core(cfg, pose_coord, heat_score)
    poses = poses[keep_pose_inds]
    heat_score = heat_score[keep_pose_inds]

    if len(keep_pose_inds) > cfg.DATASET.MAX_NUM_PEOPLE:
        heat_score, topk_inds = torch.topk(heat_score, cfg.DATASET.MAX_NUM_PEOPLE)
        poses = poses[topk_inds]

    poses = [poses.numpy()]
    scores = [i[:, 2].mean() for i in poses[0]]

    return poses, scores
