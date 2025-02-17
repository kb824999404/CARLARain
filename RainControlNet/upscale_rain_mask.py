import torch

from torch.utils.data import DataLoader
from dataset import CRIGDatasetFolder
from cldm.model import create_model, load_state_dict
from ldm.util import log_txt_as_img
from omegaconf import OmegaConf

import os
import argparse

import numpy as np
from PIL import Image
import time

from tqdm import tqdm
import random

def get_args():
    parser = argparse.ArgumentParser(description='ControlNet Predict')
    parser.add_argument('-c', '--config',
                        type=str,
                        default=None,
                        help='The config file')    
    
    args = parser.parse_args()

    if args.config:
        assert os.path.exists(args.config), ("The config file is missing.", args.config)
        cfg = OmegaConf.load(args.config)
        return cfg

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
                                            ddim_steps=args.controlnet.steps, eta=0.0,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc_full,
                                            )
        x_samples_cfg = model.decode_first_stage(samples_cfg)
        log["samples"] = x_samples_cfg

    return log

if __name__=="__main__":
    # Configs
    args = get_args()

    dataRoot = args.scene.dataRoot
    seqName = args.scene.name
    seqRoot = os.path.join(dataRoot,seqName)

    outdir = os.path.join(seqRoot,"rain_controlnet_{}".format(args.controlnet.resolution))
    os.makedirs(outdir,exist_ok=True)

    # Set GPU ID
    if args.controlnet.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.controlnet.gpu}'

    # Set random seed
    if "seed" in args.controlnet:
        set_random_seed(args.controlnet.seed)
    else:
        set_random_seed(int(time.time()))

    sd_locked = True
    only_mid_control = False

    # Create Model
    model = create_model(args.controlnet.model_config).cpu()
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Load checkpoint
    pretrained_weights = load_state_dict(args.controlnet.resume)
    model.load_state_dict(pretrained_weights)
    model = model.cuda()
    model.eval()
    

    # Load Data
    data_root = os.path.join(seqRoot,args.controlnet.source_path)
    if "max_len" in args.controlnet:
        dataset = CRIGDatasetFolder(data_root,max_len=args.controlnet.max_len,resolution=args.controlnet.resolution)
    else:
        dataset = CRIGDatasetFolder(data_root,resolution=args.controlnet.resolution)
    dataFilter = []
    # Filter data no sample
    if "filter_data" in args.controlnet and args.controlnet.filter_data:
        for item in dataset.data:
            fileName = '-'.join(item['rgb'].split('/')[1:]) 
            filePath = os.path.join(outdir,fileName)
            if not os.path.exists(filePath):
                dataFilter.append(item)
        dataset.data = dataFilter
    print("Dataset Size:",len(dataset))
    if "start_index" in args.controlnet and args.controlnet.start_index:
        dataset.data = dataset.data[args.controlnet.start_index*args.controlnet.batch_size:]
    if "end_index" in args.controlnet and args.controlnet.end_index:
        dataset.data = dataset.data[:(args.controlnet.end_index-args.controlnet.start_index)*args.controlnet.batch_size]
    print("Infer Dataset Size:",len(dataset))
    dataloader = DataLoader(dataset, num_workers=0, batch_size=args.controlnet.batch_size, shuffle=False)

    # Inference
    with torch.no_grad():
        batch_idx = 0
        for batch in tqdm(dataloader):
            batch_idx += 1
                
            images = log_images(model,batch,args.controlnet.batch_size,args.controlnet.cfg)
            key = "samples"
            images[key] = images[key].detach().cpu()
            images[key] = torch.clamp(images[key], -1., 1.)
            images[key] = (images[key] + 1.0) / 2.0
            images[key] = (images[key].numpy() * 255).astype(np.uint8)
            images[key] = images[key].transpose(0,2,3,1)
            N = images[key].shape[0]
            for i in range(N):
                fileName = batch['fileName'][i]
                Image.fromarray(images[key][i]).save(os.path.join(outdir,fileName))