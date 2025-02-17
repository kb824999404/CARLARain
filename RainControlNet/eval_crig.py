import torch

from torch.utils.data import DataLoader
from dataset import CRIGDatasetFolder
from cldm.model import create_model, load_state_dict
from ldm.util import log_txt_as_img
from einops import repeat

import os
import argparse

import numpy as np
from PIL import Image

from tqdm import tqdm
import random

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to inference Rain ControlNet."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="task",
        help="Task Name.",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./models/cldm_v15.yaml",
        help="The model config file.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/zhoukaibin/data/datasets/COCO-Stuff/dataset",
        help="Dataset root.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint.",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=16,
        help="The batch size.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution of original image.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="The max len of dataset.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs",
        help="The output dir.",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=9,
        help="The unconditional guidance scale.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="The sample steps.",
    )

    parser.add_argument("--filter_data",
        action="store_true",
        help="Filter data that has been inferred.")
    

    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="The start batch index.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="The end batch index.",
    )

    args = parser.parse_args()
    return args

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log_images(model, batch, N, unconditional_guidance_scale):
    log = dict()
    z, c = model.get_input(batch, model.first_stage_key, bs=N)
    c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
    N = min(z.shape[0], N)
    log["reconstruction"] = model.decode_first_stage(z)
    log["control"] = c_cat * 2.0 - 1.0
    log["conditioning"] = log_txt_as_img((512, 512), batch[model.cond_stage_key], size=16)


    if unconditional_guidance_scale > 1.0:
        uc_cross = model.get_unconditional_conditioning(N)
        uc_cat = c_cat  # torch.zeros_like(c_cat)
        uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}

        samples_cfg, _ = model.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                            batch_size=N, ddim=True,
                                            ddim_steps=args.steps, eta=0.0,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc_full,
                                            )
        x_samples_cfg = model.decode_first_stage(samples_cfg)
        log["samples"] = x_samples_cfg

    return log

if __name__=="__main__":
    # Set random seed
    set_random_seed(12345)

    # Configs
    args = parse_args()

    sd_locked = True
    only_mid_control = False

    # Create Model
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Load checkpoint
    pretrained_weights = load_state_dict(args.resume)
    model.load_state_dict(pretrained_weights)
    model = model.cuda()
    model.eval()
    
    task_dir = os.path.join(args.out_dir,args.task)
    os.makedirs(args.out_dir,exist_ok=True)
    os.makedirs(task_dir,exist_ok=True)

    with open(os.path.join(task_dir,"args.json"),"w") as f:
        # Save configs as file
        import json
        json.dump(vars(args),f,indent=4)

    # Load Data
    dataset = CRIGDatasetFolder(args.data_root,max_len=args.max_len,resolution=args.resolution)
    dataFilter = []
    # Filter data no sample
    if args.filter_data:
        for item in dataset.data:
            fileName = '-'.join(item['rgb'].split('/')[1:]) 
            filePath = os.path.join(task_dir,"samples",fileName)
            if not os.path.exists(filePath):
                dataFilter.append(item)
        dataset.data = dataFilter
    print("Dataset Size:",len(dataset))
    if args.start_index:
        dataset.data = dataset.data[args.start_index*args.batch_size:]
    if args.end_index:
        dataset.data = dataset.data[:(args.end_index-args.start_index)*args.batch_size]
    print("Infer Dataset Size:",len(dataset))
    dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

    # Inference
    with torch.no_grad():
        batch_idx = 0
        for batch in tqdm(dataloader):
            batch_idx += 1
                
            images = log_images(model,batch,args.batch_size,args.cfg)
            key = "samples"
            images[key] = images[key].detach().cpu()
            images[key] = torch.clamp(images[key], -1., 1.)
            images[key] = (images[key] + 1.0) / 2.0
            images[key] = (images[key].numpy() * 255).astype(np.uint8)
            images[key] = images[key].transpose(0,2,3,1)
            N = images[key].shape[0]
            for i in range(N):
                fileName = batch['fileName'][i]
                Image.fromarray(images[key][i]).save(os.path.join(task_dir,fileName))