# Convert Semantic Segmentation Data to Cityscapes format
import argparse
import os
import yaml
from config import Config

import mmcv
import numpy  as np
import cv2

from cityscapesscripts.helpers.labels import labels

color2trainId = { label.color: label.trainId for label in labels  }
color2Id = { label.color: label.id for label in labels  }
color2Id[(0,0,142)] = 0

# 将调色板字典转换为一个查找表
max_color_value = 256
lookup_table_train = np.full((max_color_value, max_color_value, max_color_value), 255, dtype=np.uint8)
lookup_table = np.full((max_color_value, max_color_value, max_color_value), 0, dtype=np.uint8)
for rgb, gray in color2trainId.items():
    lookup_table_train[rgb] = gray
for rgb, gray in color2Id.items():
    lookup_table[rgb] = gray

    
# Read config from args and yaml file
def get_args():
    parser = argparse.ArgumentParser(description='CARLA Run')

    parser.add_argument('-c', '--config',
                        type=str,
                        default=None,
                        help='The config file')    

    args = parser.parse_args()

    if args.config:
        assert os.path.exists(args.config), ("The config file is missing.", args.config)
        with open(args.config,"r") as f:
            cfg = yaml.safe_load(f)['scene']
        for key in cfg:
            args.__dict__[key] = cfg[key]

    return args


# RGB语义分割图转灰度图
def rgb_to_mask(x,is_train=False):
    mask = np.zeros((x.shape[0], x.shape[1]), dtype=np.uint8)
    
    # 使用查找表将RGB图像转换为灰度图像
    if is_train:
        mask = lookup_table_train[x[..., 0], x[..., 1], x[..., 2]]
    else:
        mask = lookup_table[x[..., 0], x[..., 1], x[..., 2]]

    return mask

def convert_rgb_to_label_eval(fileName):
    source_file = os.path.join(gt_dir,fileName)
    out_file = os.path.join(out_dir_eval,fileName)

    source = cv2.imread(source_file, cv2.IMREAD_COLOR)[:,:,::-1]
    out = rgb_to_mask(source,is_train=False)
    cv2.imwrite(out_file,out)

def convert_rgb_to_label_train(fileName):
    source_file = os.path.join(gt_dir,fileName)
    out_file = os.path.join(out_dir_train,fileName)

    source = cv2.imread(source_file, cv2.IMREAD_COLOR)[:,:,::-1]
    out = rgb_to_mask(source,is_train=True)
    cv2.imwrite(out_file,out)


if __name__ == '__main__':
    args = get_args()
    nproc = 20
    seqRoot = os.path.join(args.dataRoot,args.name)
    print(seqRoot)
    
    gt_dir = os.path.join(seqRoot,Config.cameraPaths["ss"])
    out_dir_train = os.path.join(seqRoot,Config.cameraPaths["ss"]+"_train")
    out_dir_eval = os.path.join(seqRoot,Config.cameraPaths["ss"]+"_eval")

    gt_files = [ file for file in mmcv.scandir(gt_dir, '.png') ]

    # mmcv.mkdir_or_exist(out_dir_train)
    # mmcv.track_parallel_progress(convert_rgb_to_label_train, gt_files,
    #                                 nproc)

    mmcv.mkdir_or_exist(out_dir_eval)
    mmcv.track_parallel_progress(convert_rgb_to_label_eval, gt_files,
                                    nproc)