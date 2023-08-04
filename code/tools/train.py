from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import sys
script_path = os.path.abspath(__file__)
directory_path = os.path.dirname(os.path.dirname(script_path))
sys.path.append(directory_path)
import pprint
import shutil
import warnings
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
import lib.models as models
from lib.config import cfg
from lib.config import update_config
from lib.core.loss import MultiLossFactory
from lib.core.trainer_gc import do_train
from lib.dataset import make_dataloader, make_val_dataloader
from lib.fp16_utils.fp16util import network_to_half
from lib.fp16_utils.fp16_optimizer import FP16_Optimizer
from lib.utils.utils import create_logger
from lib.utils.utils import get_optimizer_frozen
from lib.utils.utils import save_checkpoint
from lib.utils.utils import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    # distributed training
    parser.add_argument('--gpu', help='gpu id for multiprocessing training', type=str)
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23458', type=str, help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    update_config(cfg, args)

    cfg.defrost()
    cfg.RANK = args.rank
    cfg.freeze()

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or cfg.MULTIPROCESSING_DISTRIBUTED

    ngpus_per_node = torch.cuda.device_count()
    if cfg.MULTIPROCESSING_DISTRIBUTED:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, final_output_dir, tb_log_dir))
    else:
        main_worker(','.join([str(i) for i in cfg.GPUS]), ngpus_per_node, args, final_output_dir, tb_log_dir)


def main_worker(gpu, ngpus_per_node, args, final_output_dir, tb_log_dir):
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.FP16.ENABLED:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if cfg.FP16.STATIC_LOSS_SCALE != 1.0:
        if not cfg.FP16.ENABLED:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if cfg.MULTIPROCESSING_DISTRIBUTED:
            args.rank = args.rank * ngpus_per_node + gpu
        print('Init process group: dist_url: {}, world_size: {}, rank: {}'. format(args.dist_url, args.world_size, args.rank))
        dist.init_process_group(
            backend=cfg.DIST_BACKEND,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank)
    update_config(cfg, args)

    # setup logger
    logger, _ = setup_logger(final_output_dir, args.rank, 'train')
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=True)

    # Freezee pretrained
    ct = 0
    list_name_frozen = []
    for name, param in model.named_parameters():
        ct += 1
        # 481 for small and 916 for full
        if (ct < 916):
            # if(ct<924 or 'transition_heatmap' in name):
            list_name_frozen.append(name)

    # copy model file
    if not cfg.MULTIPROCESSING_DISTRIBUTED or (cfg.MULTIPROCESSING_DISTRIBUTED and args.rank % ngpus_per_node == 0):
        this_dir = os.path.dirname(__file__)
        shutil.copy2(os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'), final_output_dir)

    writer_dict = {'writer': SummaryWriter(logdir=tb_log_dir), 'train_global_steps': 0, 'valid_global_steps': 0,}

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.MODEL.SYNC_BN and not args.distributed:
        print('Warning: Sync BatchNorm is only supported in distributed training.')

    if args.distributed:
        if cfg.MODEL.SYNC_BN:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per DistributedDataParallel, we need to divide the batch size ourselves based on the total number of GPUs we have
            # args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    loss_factory = MultiLossFactory(cfg).cuda()

    # Data loading code
    train_loader = make_dataloader(cfg, is_train=True, distributed=args.distributed)
    logger.info(train_loader.dataset)

    val_loader = make_val_dataloader(cfg)

    best_perf = -1
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer_frozen(cfg, model, list_name_frozen)
    #     optimizer = get_optimizer(cfg, model)

    if cfg.FP16.ENABLED:
        optimizer = FP16_Optimizer(optimizer,
            static_loss_scale=cfg.FP16.STATIC_LOSS_SCALE,
            dynamic_loss_scale=cfg.FP16.DYNAMIC_LOSS_SCALE)

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth.tar')
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

    if cfg.FP16.ENABLED:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer.optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch)

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # train one epoch
        do_train(cfg, model, train_loader, loss_factory, optimizer, epoch, final_output_dir, tb_log_dir, writer_dict, fp16=cfg.FP16.ENABLED)
        lr_scheduler.step()

        perf_indicator = epoch
        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False
        if not cfg.MULTIPROCESSING_DISTRIBUTED or (cfg.MULTIPROCESSING_DISTRIBUTED and args.rank == 0):

            # model.eval()
            # do_val(cfg, model, val_loader, loss_factory, epoch, final_output_dir, tb_log_dir, writer_dict)

            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir, 'final_state{}.pth.tar'.format(gpu))

    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()