from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import torch

from lib.utils.utils import AverageMeter

def hierarchical_pool(heatmap):
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

def get_maximum_from_heatmap(cfg, heatmap):
    print(heatmap.shape)
    maxm = hierarchical_pool(heatmap)
    maxm = torch.eq(maxm, heatmap).float()
    heatmap = heatmap * maxm
    hh  = heatmap[0][0]
    print(hh.shape)
    for i in range (hh.shape[0]):
        for j in range (hh.shape[1]):
            if hh[i][j] != 0:
                print((i,j), hh[i][j])

    nonzero_idx = torch.nonzero(hh)

    coordinates = nonzero_idx.tolist()

    print(coordinates)

    scores = heatmap.view(-1)
    scores, pos_ind = scores.topk(cfg.DATASET.MAX_NUM_PEOPLE)

    select_ind = (scores > (cfg.TEST.KEYPOINT_THRESHOLD)).nonzero()
    scores = scores[select_ind][:, 0]
    pos_ind = pos_ind[select_ind][:, 0]

    return pos_ind, scores


def do_train(cfg, model, data_loader, loss_factory, optimizer, epoch, output_dir, tb_log_dir, writer_dict, fp16=False):
    logger = logging.getLogger("Training")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmap_loss_meter = AverageMeter()
    center_loss_meter = AverageMeter()
    offset_loss_init_meter = AverageMeter()
    offset_loss_meter = AverageMeter()
    visibility_loss_meter = AverageMeter()
    feature_loss_meter = AverageMeter()

    model.train()

    end = time.time()
    for i, (image, heatmap, mask, offset, offset_w, visibility_map) in enumerate(data_loader):

        data_time.update(time.time() - end)

        pheatmap_init, poffset_init, poffset_final, pvisibility, features = model(image)

        pheatmap_init = pheatmap_init.float()
        poffset_init = poffset_init.float()
        # pcenter_final = pcenter_final.float()
        poffset_final = poffset_final.float()
        # pheatmap_fusion = pheatmap_fusion.float()
        # poffset_fusion = poffset_fusion.float()
        pvisibility = pvisibility.float()
        features = features.float()

        heatmap = heatmap.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        offset = offset.cuda(non_blocking=True)
        offset_w = offset_w.cuda(non_blocking=True)
        visibility_map = visibility_map.cuda(non_blocking=True)
        center_heatmap = heatmap[:, -1:, :, :]

        center_mask = mask[:, -1:, :, :]

        heatmap_loss_init, offset_loss_init, offset_loss_final, visibility_loss, feature_loss = \
            loss_factory(pheatmap_init, poffset_init, poffset_final, pvisibility, heatmap, center_mask, mask, offset, offset_w, visibility_map, features, center_heatmap)

        loss = 0
        if heatmap_loss_init is not None:
            heatmap_loss_meter.update(heatmap_loss_init.item(), image.size(0))
            loss = loss + heatmap_loss_init

        if offset_loss_init is not None:
            offset_loss_init_meter.update(offset_loss_init.item(), image.size(0))
            loss = loss + offset_loss_init
        if offset_loss_final is not None:
            offset_loss_meter.update(offset_loss_final.item(), image.size(0))
            loss = loss + offset_loss_final

        if visibility_loss is not None:
            visibility_loss_meter.update(visibility_loss.item(), image.size(0))
            loss = loss + visibility_loss
        # if feature_loss is not None:
        #     feature_loss_meter.update(feature_loss.item(), image.size(0))
        #     loss = loss + feature_loss

        optimizer.zero_grad()
        if fp16:
            optimizer.backward(loss)
            optimizer.clip_master_grads(0.1, norm_type=2)
        else:
            loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.PRINT_FREQ == 0 and cfg.RANK == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  '{heatmaps_loss}{offset_loss_init}{offset_loss}{visibility_loss}'.format(
                epoch, i, len(data_loader),
                batch_time=batch_time,
                speed=image.size(0) / batch_time.val,
                data_time=data_time,
                heatmaps_loss=_get_loss_info(heatmap_loss_meter, 'heatmaps'),
                offset_loss_init=_get_loss_info(offset_loss_init_meter, 'offset_init'),
                offset_loss=_get_loss_info(offset_loss_meter, 'offset'),
                visibility_loss=_get_loss_info(visibility_loss_meter, 'visibility')
            )
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar(
                'train_heatmap_loss',
                heatmap_loss_meter.val,
                global_steps
            )
            writer.add_scalar(
                'train_offset_init_loss',
                offset_loss_init_meter.val,
                global_steps
            )
            writer.add_scalar(
                'train_offset_loss',
                offset_loss_meter.val,
                global_steps
            )
            writer.add_scalar(
                'train_visibility_loss',
                visibility_loss_meter.val,
                global_steps
            )
            writer_dict['train_global_steps'] = global_steps + 1


def do_val(cfg, model, data_loader, loss_factory, epoch, output_dir, tb_log_dir, writer_dict):
    logger = logging.getLogger("Validation")

    batch_time = AverageMeter()
    data_time = AverageMeter()

    heatmap_loss_meter_val = AverageMeter()
    offset_loss_init_meter_val = AverageMeter()
    offset_loss_meter_val = AverageMeter()
    visibility_loss_meter_val = AverageMeter()

    end = time.time()

    model.eval()
    with torch.no_grad():
        for i, (image, heatmap, mask, offset, offset_w, visibility_map) in enumerate(data_loader):

            data_time.update(time.time() - end)

            pheatmap, poffset_init, poffset, pvisibility = model(image)

            heatmap = heatmap.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            offset = offset.cuda(non_blocking=True)
            offset_w = offset_w.cuda(non_blocking=True)
            visibility_map = visibility_map.cuda(non_blocking=True)

            heatmap_loss, offset_loss_init, offset_loss, visibility_loss = \
                loss_factory(pheatmap, poffset_init, heatmap, mask, offset, offset_w, visibility_map, poffset,
                             pvisibility)

            loss = 0
            if heatmap_loss is not None:
                heatmap_loss_meter_val.update(heatmap_loss.item(), image.size(0))
                loss = loss + heatmap_loss
            if offset_loss_init is not None:
                offset_loss_init_meter_val.update(offset_loss_init.item(), image.size(0))
                loss = loss + offset_loss_init
            if offset_loss is not None:
                offset_loss_meter_val.update(offset_loss.item(), image.size(0))
                loss = loss + offset_loss
            if visibility_loss is not None:
                visibility_loss_meter_val.update(visibility_loss.item(), image.size(0))
                loss = loss + visibility_loss

            batch_time.update(time.time() - end)
            end = time.time()

            if i == len(data_loader) - 1:
                msg = 'Epoch_Val: [{0}][{1}/{2}]\t' \
                      'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed: {speed:.1f} samples/s\t' \
                      'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      '{heatmaps_loss}{offset_loss_init}{offset_loss}{visibility_loss}'.format(
                    epoch, i, len(data_loader),
                    batch_time=batch_time,
                    speed=image.size(0) / batch_time.val,
                    data_time=data_time,
                    heatmaps_loss=_get_loss_info(heatmap_loss_meter_val, 'heatmaps'),
                    offset_loss_init=_get_loss_info(offset_loss_init_meter_val, 'offset_init'),
                    offset_loss=_get_loss_info(offset_loss_meter_val, 'offset'),
                    visibility_loss=_get_loss_info(visibility_loss_meter_val, 'visibility')
                )
                logger.info(msg)

                writer = writer_dict['writer']
                global_steps = writer_dict['valid_global_steps']
                writer.add_scalar(
                    'train_heatmap_loss',
                    heatmap_loss_meter_val.val,
                    global_steps
                )
                writer.add_scalar(
                    'train_offset_init_loss',
                    offset_loss_init_meter_val.val,
                    global_steps
                )
                writer.add_scalar(
                    'train_offset_loss',
                    offset_loss_meter_val.val,
                    global_steps
                )
                writer.add_scalar(
                    'train_visibility_loss',
                    visibility_loss_meter_val.val,
                    global_steps
                )
                writer_dict['valid_global_steps'] = global_steps + 1


def _get_loss_info(meter, loss_name):
    msg = ''
    msg += '{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(
        name=loss_name, meter=meter
    )

    return msg
