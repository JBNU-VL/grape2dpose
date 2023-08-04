
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data

from .COCODataset import CocoDataset as coco
from .COCODataset import CocoRescoreDataset as rescore_coco
from .COCOKeypoints import CocoKeypoints as coco_kpt
from .CrowdPoseDataset import CrowdPoseDataset as crowd_pose
from .CrowdPoseDataset import CrowdPoseRescoreDataset as rescore_crowdpose
from .CrowdPoseKeypoints import CrowdPoseKeypoints as crowd_pose_kpt
from .OCHUMANDataset import OchumanDataset as ochuman
from .OCHUMANDataset import OchumanRescoreDataset as rescore_ochuman
from .OCHUMANKeypoints import OchumanKeypoints as ochuman_kpt
from .transforms import build_transforms, build_transforms_val
from .target_generators import HeatmapGenerator
from .target_generators import OffsetGenerator


def build_dataset(cfg, is_train):
    assert is_train is True, 'Please only use build_dataset for training.'

    transforms = build_transforms(cfg, is_train)

    heatmap_generator = HeatmapGenerator(
        cfg.DATASET.OUTPUT_SIZE, cfg.DATASET.NUM_JOINTS
    )
    offset_generator = OffsetGenerator(
        cfg.DATASET.OUTPUT_SIZE, cfg.DATASET.OUTPUT_SIZE,
        cfg.DATASET.NUM_JOINTS, cfg.DATASET.OFFSET_RADIUS
    ) 

    dataset = eval(cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.TRAIN,
        heatmap_generator,
        offset_generator,
        transforms
    )

    return dataset


def build_val_dataset(cfg):
    transforms = build_transforms_val(cfg)

    heatmap_generator = HeatmapGenerator(
        cfg.DATASET.OUTPUT_SIZE, cfg.DATASET.NUM_JOINTS
    )
    offset_generator = OffsetGenerator(
        cfg.DATASET.OUTPUT_SIZE, cfg.DATASET.OUTPUT_SIZE,
        cfg.DATASET.NUM_JOINTS, cfg.DATASET.OFFSET_RADIUS
    )

    dataset = eval(cfg.DATASET.DATASET)(
        cfg,
        cfg.DATASET.TEST,
        heatmap_generator,
        offset_generator,
        transforms
    )

    return dataset

def make_dataloader(cfg, is_train=True, distributed=False):
    if is_train:
        images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
        shuffle = True
    else:
        images_per_gpu = cfg.TEST.IMAGES_PER_GPU
        shuffle = False
    images_per_batch = images_per_gpu * len(cfg.GPUS)
    dataset = build_dataset(cfg, is_train)

    if is_train and distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False
    else:
        train_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_batch,
        shuffle=shuffle,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=train_sampler
    )

    return data_loader


def make_test_dataloader(cfg):
    dataset = eval(cfg.DATASET.DATASET_TEST)(
        cfg, cfg.DATASET.TEST
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return data_loader, dataset

def make_val_dataloader(cfg):
    images_per_gpu = cfg.TEST.IMAGES_PER_GPU
    shuffle = False

    dataset = build_val_dataset(cfg)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_gpu,
        shuffle=shuffle,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    return data_loader
