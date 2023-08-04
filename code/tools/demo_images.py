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
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
import lib.models as models
from lib.config import cfg
from lib.config import update_config
from lib.core.inference_attention import get_multi_stage_outputs
from lib.core.inference_attention import aggregate_results
from lib.core.nms_attention_newscore import pose_nms
from lib.core.match_attention import match_pose_to_heatmap
from lib.utils.utils import create_logger
from lib.utils.transforms import resize_align_multi_scale
from lib.utils.transforms import get_final_preds
from lib.utils.transforms import get_multi_scale_size
from lib.fp16_utils.fp16util import network_to_half
import cv2
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info('| Arch ' +  ' '.join(['| {}'.format(name) for name in names]) +  ' |' )
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info('| ' + full_arch_name + ' ' + ' '.join(['| {:.3f}'.format(value) for value in values]) + ' |')


def show_crowdpose_skeleton(img, kpts, color=(255,128,128), thr=0.2):
    kpts = np.array(kpts).reshape(-1,3)
    skelenton = [[0, 2], [1, 3], [2, 4], [3, 5], [6, 8], [8, 10], [7, 9], [9, 11], [12, 13], [0, 13], [1, 13], [6,13],[7, 13]]
    points_num = [num for num in range(14)]
    for sk in skelenton:
        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1] , 1]))

        if pos1[0] > 0 and pos1[1] > 0 and pos2[0] > 0 and pos2[1] > 0 and kpts[sk[0], 2] > thr and kpts[sk[1], 2] > thr:
            cv2.line(img, pos1, pos2, color, 4, 8)
    for points in points_num:
        pos = (int(kpts[points,0]),int(kpts[points,1]))
        if pos[0] > 0 and pos[1] > 0 and kpts[points,2] > thr:
            cv2.circle(img, pos, 4, (0, 0, 0), 2)
    return img

def show_crowdpose_center_offset(img, kpts, color=(255, 128, 128), thr=0.2):
    kpts = np.array(kpts).reshape(-1, 3)
    points_num = [num for num in range(14)]
    center = np.mean(kpts[points_num, :2], axis=0, dtype=np.int32)
    for points in points_num:
        pos = (int(kpts[points, 0]), int(kpts[points, 1]))
        if pos[0] > 0 and pos[1] > 0 and kpts[points, 2] > thr:
            cv2.line(img, center, pos, color, 4, 8)
            cv2.circle(img, pos, 8, (0, 255, 0), 8)
            cv2.circle(img, center, 4, (0, 0, 0), 4)
    return img


def main():
    args = parse_args()
    update_config(cfg, args)
    logger, final_output_dir, _ = create_logger(cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.grape_model.get_pose_net')(cfg, is_train=False)

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(final_output_dir, 'model_best.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    all_reg_preds = []
    all_reg_scores = []

    MATCH_HMP = True

    image_list = []

    image_file_or_path = '../../demo_images_video/images/'
    for filename in os.listdir(image_file_or_path):
        if filename.endswith(".png") or filename.endswith(".jpg") and 'pred' not in filename:
            image_list.append(image_file_or_path + filename)

    for image_file in image_list:
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_draw = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        base_size, center, scale = get_multi_scale_size(image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0)

        model.eval()
        with torch.no_grad():
            heatmap_sum = 0
            visibilitymap_sum = 0
            poses = []

            scales = cfg.TEST.SCALE_FACTOR

            ori_h, ori_w, _ = image.shape
            if (ori_h < ori_w and ori_w / ori_h > 2):
                scales = [1]
            if (ori_w < ori_h and ori_h / ori_w > 2):
                scales = [1]

            for scale in sorted(scales, reverse=True):
                image_resized, center, scale_resized = resize_align_multi_scale(image, cfg.DATASET.INPUT_SIZE, scale, 1.0)
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()
                heatmap, posemap, visibilitymap = get_multi_stage_outputs(cfg, model, image_resized, cfg.TEST.FLIP_TEST)
                heatmap_sum, poses, visibilitymap_sum = aggregate_results(cfg, heatmap_sum, visibilitymap_sum, poses, heatmap, posemap, visibilitymap, scale)
            heatmap_avg = heatmap_sum / len(scales)
            visibilitymap_avg = visibilitymap_sum / len(scales)

            poses, scores = pose_nms(cfg, heatmap_avg, poses, visibilitymap_avg, scales)

            if len(scores) == 0:
                all_reg_preds.append([])
                all_reg_scores.append([])
            else:
                if MATCH_HMP:
                    poses = match_pose_to_heatmap(cfg, poses, heatmap_avg)
                else:
                    poses_no_match = torch.tensor(poses)
                    poses = [torch.squeeze(poses_no_match[:, :, :, :3]).cpu().numpy()]

                final_poses = get_final_preds(poses, center, scale_resized, base_size)

            final_results = []
            for i in range(len(scores)):
                if scores[i] > 0:
                    final_results.append(final_poses[i])

            for coords in final_results:
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                if MATCH_HMP:
                    imgshow = show_crowdpose_skeleton(image_draw, coords, color=color)
                else:
                    imgshow = show_crowdpose_center_offset(image_draw, coords, color=color)

            pose_dir = image_file[:-4] + '_pred.jpg'
            cv2.imwrite(pose_dir, imgshow)

if __name__ == '__main__':
    main()
