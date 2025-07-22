import argparse, os
from omegaconf import OmegaConf

import random, time
import numpy as np
import torch
import cv2
from PIL import Image
from models.crig import CRIG_EDNet_64, CRIG_EDNet_128
from models.crig import CRIG_EDNet_Resnet_64, CRIG_EDNet_Transformer_64, CRIG_EDNet_Resnet_128, CRIG_EDNet_Transformer_128
from models.crig import CRIG_FastGAN_64, CRIG_FastGAN_128, CRIG_FastGAN_256, CRIG_FastGAN_512


configFile = "/home/ubuntu/Code/CARLARain/configs/seqFrontend.yaml"


def saveTensorAsImg(T,path):
    T = torch.clamp(T, 0., 1.)
    I = T.data.cpu().numpy()
    I = I.transpose(0, 2, 3, 1)
    I = I[0]*255
    Image.fromarray(I.astype(np.uint8)).save(path)


def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def predict_rain_from_zero(out_path,intensity,direction,randomSeed):
    opt = OmegaConf.load(configFile)


    # Build model
    print('Loading CRIGNet model ...')
    if opt.crig.patchSize == 64:
        if opt.crig.backbone == "Resnet":
            netED = CRIG_EDNet_Resnet_64(opt.crig.nc, opt.crig.nz, opt.crig.nef, opt.crig.nlabel).cuda()
        elif opt.crig.backbone == "Transformer":
            netED = CRIG_EDNet_Transformer_64(opt.crig.nc, opt.crig.nz, opt.crig.nef, opt.crig.nlabel).cuda()
        elif opt.crig.backbone == "FastGAN":
            netED = CRIG_FastGAN_64(opt.crig.nc, opt.crig.nz, opt.crig.nef, opt.crig.nlabel).cuda()
        else:
            netED = CRIG_EDNet_64(opt.crig.nc, opt.crig.nz, opt.crig.nef, opt.crig.nlabel).cuda()
    elif opt.crig.patchSize == 128:
        if opt.crig.backbone == "Resnet":
            netED = CRIG_EDNet_Resnet_128(opt.crig.nc, opt.crig.nz, opt.crig.nef, opt.crig.nlabel, w_dim=opt.crig.w_dim, use_mapping=opt.crig.use_mapping).cuda()
        elif opt.crig.backbone == "Transformer":
            netED = CRIG_EDNet_Transformer_128(opt.crig.nc, opt.crig.nz, opt.crig.nef, opt.crig.nlabel, w_dim=opt.crig.w_dim, use_mapping=opt.crig.use_mapping).cuda()
        elif opt.crig.backbone == "FastGAN":
            netED = CRIG_FastGAN_128(opt.crig.nc, opt.crig.nz, opt.crig.nef, opt.crig.nlabel, w_dim=opt.crig.w_dim, use_mapping=opt.crig.use_mapping).cuda()
        else:
            netED = CRIG_EDNet_128(opt.crig.nc, opt.crig.nz, opt.crig.nef, opt.crig.nlabel, w_dim=opt.crig.w_dim, use_mapping=opt.crig.use_mapping).cuda()
    elif opt.crig.patchSize == 256:
        if opt.crig.backbone == "FastGAN":
            netED = CRIG_FastGAN_256(opt.crig.nc, opt.crig.nz, opt.crig.nef, opt.crig.nlabel, w_dim=opt.crig.w_dim, use_mapping=opt.crig.use_mapping).cuda()
        else:
            print("Only FastGAN backbone for patch size 256!")
    elif opt.crig.patchSize == 512:
        if opt.crig.backbone == "FastGAN":
            netED = CRIG_FastGAN_512(opt.crig.nc, opt.crig.nz, opt.crig.nef, opt.crig.nlabel, w_dim=opt.crig.w_dim, use_mapping=opt.crig.use_mapping).cuda()
        else:
            print("Only FastGAN backbone for patch size 256!")   
    else:
        print("Please check patch size, must be 64/128/256")
    netED.load_state_dict(torch.load(opt.crig.netED))
    print("Loading CRIGNet model done.")

    seed_everything(randomSeed)
    
    label = torch.tensor(np.array([[intensity,direction]]),dtype=torch.float32).cuda()
    z_random = torch.randn(1, opt.crig.nz).cuda()
    with torch.no_grad():  #
        if opt.crig.use_mapping:
            R = netED.sample_mapping(z_random,label)
        else:
            R = netED.sample(torch.cat([z_random,label],dim=1))
        R = torch.clamp(R, 0., 1.)
        saveTensorAsImg(R,out_path)
        print(f"Save to {out_path}")

    print("Generate Rain Image Successfully!")

    del netED
    torch.cuda.empty_cache()
