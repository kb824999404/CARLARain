# "
# This is the latent space  disentanglement experiment. Taking the generator trained on rain100L as example
# "
import os
import argparse
import glob
import numpy as np
import torch
import cv2
import json
from PIL import Image
from natsort import natsorted
from models.crig import CRIG_EDNet_64, CRIG_EDNet_128
from models.crig import CRIG_EDNet_Resnet_64, CRIG_EDNet_Transformer_64, CRIG_EDNet_Resnet_128, CRIG_EDNet_Transformer_128
from models.crig import CRIG_FastGAN_64, CRIG_FastGAN_128, CRIG_FastGAN_256, CRIG_FastGAN_512
import random, time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--nc', type=int, default=3, help='size of the RGB image')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--nef', type=int, default=32)
parser.add_argument('--nlabel', type=int, default=2, help='size of the render label vector')
# Backbone
parser.add_argument("--backbone", type=str, default="VRGNet", help='The backbone of generator')
parser.add_argument("--use_mapping", type=str2bool, default=False, help='use mapping network for label')
parser.add_argument('--w_dim', type=int, default=128, help='size of the latent w vector')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument('--netED', default='./syn100lmodels/ED_state_700.pt', help="path to netED for z--rain display")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--patchSize', type=int, default=64, help='the height / width of the input image to network')
# Dataset
parser.add_argument("--result_path", type=str, default="./result_code", help='folder of input images')

opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


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

def main():
    # Build model
    print('Loading model ...\n')
    if opt.patchSize == 64:
        if opt.backbone == "Resnet":
            netED = CRIG_EDNet_Resnet_64(opt.nc, opt.nz, opt.nef, opt.nlabel).cuda()
        elif opt.backbone == "Transformer":
            netED = CRIG_EDNet_Transformer_64(opt.nc, opt.nz, opt.nef, opt.nlabel).cuda()
        elif opt.backbone == "FastGAN":
            netED = CRIG_FastGAN_64(opt.nc, opt.nz, opt.nef, opt.nlabel).cuda()
        else:
            netED = CRIG_EDNet_64(opt.nc, opt.nz, opt.nef, opt.nlabel).cuda()
    elif opt.patchSize == 128:
        if opt.backbone == "Resnet":
            netED = CRIG_EDNet_Resnet_128(opt.nc, opt.nz, opt.nef, opt.nlabel, w_dim=opt.w_dim, use_mapping=opt.use_mapping).cuda()
        elif opt.backbone == "Transformer":
            netED = CRIG_EDNet_Transformer_128(opt.nc, opt.nz, opt.nef, opt.nlabel, w_dim=opt.w_dim, use_mapping=opt.use_mapping).cuda()
        elif opt.backbone == "FastGAN":
            netED = CRIG_FastGAN_128(opt.nc, opt.nz, opt.nef, opt.nlabel, w_dim=opt.w_dim, use_mapping=opt.use_mapping).cuda()
        else:
            netED = CRIG_EDNet_128(opt.nc, opt.nz, opt.nef, opt.nlabel, w_dim=opt.w_dim, use_mapping=opt.use_mapping).cuda()
    elif opt.patchSize == 256:
        if opt.backbone == "FastGAN":
            netED = CRIG_FastGAN_256(opt.nc, opt.nz, opt.nef, opt.nlabel, w_dim=opt.w_dim, use_mapping=opt.use_mapping).cuda()
        else:
            print("Only FastGAN backbone for patch size 256!")
    elif opt.patchSize == 512:
        if opt.backbone == "FastGAN":
            netED = CRIG_FastGAN_512(opt.nc, opt.nz, opt.nef, opt.nlabel, w_dim=opt.w_dim, use_mapping=opt.use_mapping).cuda()
        else:
            print("Only FastGAN backbone for patch size 256!")   
    else:
        print("Please check patch size, must be 64/128/256")
    netED.load_state_dict(torch.load(opt.netED))

    os.makedirs(opt.result_path,exist_ok=True)


    label_origin = torch.randn(1, opt.nlabel).cuda()
    z_origin = torch.randn(1, opt.nz).cuda()

    # Disentanglement For Label
    interpolation = np.arange(0, 1+0.01, 0.1)
    for label0 in interpolation:
        for label1 in interpolation:
            print(label0,label1)
            curr_label = torch.Tensor([[label0,label1]]).cuda() 
            with torch.no_grad():  #
                if opt.use_mapping:
                    R = netED.sample_mapping(z_origin,curr_label)
                else:
                    R = netED.sample(torch.cat([z_origin,curr_label],dim=1))
                R = torch.clamp(R, 0., 1.)
                saveTensorAsImg(R,os.path.join(opt.result_path,"z_orig_I{:.2f}_W{:.2f}.jpg".format(label0,label1)))

    # Disentanglement For Z
    random.seed(time.time())
    for i in range(100):
        print("random_z:",i)
        random_z = torch.randn(1, opt.nz).cuda()
        with torch.no_grad():  #
            if opt.use_mapping:
                R = netED.sample_mapping(random_z,label_origin)
            else:
                R = netED.sample(torch.cat([random_z,label_origin],dim=1))
            R = torch.clamp(R, 0., 1.)
            saveTensorAsImg(R,os.path.join(opt.result_path,"z_random_{}.jpg".format(i)))



if __name__ == "__main__":
    main()

