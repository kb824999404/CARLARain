import argparse, os, sys, datetime, glob
from omegaconf import OmegaConf

import random, time
import numpy as np
import torch
import cv2
from PIL import Image
from models.crig import CRIG_EDNet_64, CRIG_EDNet_128
from models.crig import CRIG_EDNet_Resnet_64, CRIG_EDNet_Transformer_64, CRIG_EDNet_Resnet_128, CRIG_EDNet_Transformer_128
from models.crig import CRIG_FastGAN_64, CRIG_FastGAN_128, CRIG_FastGAN_256, CRIG_FastGAN_512
from tqdm import tqdm

import json

def get_args():
    parser = argparse.ArgumentParser(description='CRIG Predict')
    parser.add_argument('-c', '--config',
                        type=str,
                        default=None,
                        help='The config file')    
    
    args = parser.parse_args()

    if args.config:
        assert os.path.exists(args.config), ("The config file is missing.", args.config)
        cfg = OmegaConf.load(args.config)
        return cfg
    

def is_image(img_name):
    if img_name.endswith(".jpg") or img_name.endswith(".bmp") or img_name.endswith(".png"):
        return True
    else:
        return False

def saveTensorAsImg(T,path):
    T = torch.clamp(T, 0., 1.)
    I = T.data.cpu().numpy()
    I = I.transpose(0, 2, 3, 1)
    I = I[0]*255
    Image.fromarray(I.astype(np.uint8)).save(path)

def readImgAsTensor(path):
    O = cv2.imread(path)
    b, g, r = cv2.split(O)
    O = cv2.merge([r, g, b])
    O = O.astype(np.float32) / 255
    O = np.transpose(O, (2, 0, 1))
    O = O[None,:]
    O = torch.Tensor(O).cuda()
    return O

def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(opt):
    dataRoot = opt.scene.dataRoot
    seqName = opt.scene.name
    seqRoot = os.path.join(dataRoot,seqName)

    outdir = os.path.join(seqRoot,"rain_crig_{}".format(opt.crig.patchSize))

    # Build model
    print('Loading model ...\n')
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

    os.makedirs(outdir,exist_ok=True)

    data_info = []

    if "seed" in opt.crig:
        seed_everything(opt.crig.seed)
    else:
        seed_everything(int(time.time()))

    if opt.crig.mode == "RANDOM_Z_FIXED_LABEL":
        label_origin = torch.tensor(np.array([opt.crig.label]),dtype=torch.float32).cuda()
        for i in tqdm(range(opt.crig.sampleNum)):
            z_random = torch.randn(1, opt.crig.nz).cuda()
            with torch.no_grad():  #
                if opt.crig.use_mapping:
                    R = netED.sample_mapping(z_random,label_origin)
                else:
                    R = netED.sample(torch.cat([z_random,label_origin],dim=1))
                R = torch.clamp(R, 0., 1.)
                saveTensorAsImg(R,os.path.join(outdir,"rain_{}.jpg".format(i)))
            data_info.append({
                'img': "rain_{}.jpg".format(i),
                'label': label_origin.cpu().numpy().tolist()[0]
            })
    elif opt.crig.mode == "RANDOM_LABEL_FIXED_Z": 
        z_random = torch.randn(1, opt.crig.nz).cuda()
        for i in tqdm(range(opt.crig.sampleNum)):
            label_random = torch.rand(1, opt.crig.nlabel).cuda() * 2.0 - 1.0
            with torch.no_grad():  #
                if opt.crig.use_mapping:
                    R = netED.sample_mapping(z_random,label_random)
                else:
                    R = netED.sample(torch.cat([z_random,label_random],dim=1))
                R = torch.clamp(R, 0., 1.)
                saveTensorAsImg(R,os.path.join(outdir,"rain_{}.jpg".format(i)))
            data_info.append({
                'img': "rain_{}.jpg".format(i),
                'label': label_random.cpu().numpy().tolist()[0]
            })
    else:
        for i in tqdm(range(opt.crig.sampleNum)):
            label_random = torch.rand(1, opt.crig.nlabel).cuda() * 2.0 - 1.0
            z_random = torch.randn(1, opt.crig.nz).cuda()
            with torch.no_grad():  #
                if opt.crig.use_mapping:
                    R = netED.sample_mapping(z_random,label_random)
                else:
                    R = netED.sample(torch.cat([z_random,label_random],dim=1))
                R = torch.clamp(R, 0., 1.)
                saveTensorAsImg(R,os.path.join(outdir,"rain_{}.jpg".format(i)))
            data_info.append({
                'img': "rain_{}.jpg".format(i),
                'label': label_random.cpu().numpy().tolist()[0]
            })
    
    with open(os.path.join(outdir,'data_info.json'),'w') as f:
        json.dump(data_info,f)

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    opt = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{opt.crig.gpu}'
    main(opt)
