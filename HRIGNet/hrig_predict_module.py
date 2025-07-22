import argparse, os, sys, datetime, glob
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from main import instantiate_from_config
import ipdb
import yaml

configFile = "/home/ubuntu/Code/CARLARain/configs/seqFrontend.yaml"


def loadImgRGB(path):
    img = Image.open(path)
    if not img.mode == "RGB":
        img = img.convert("RGB")
    img = np.array(img).astype(np.uint8)
    return img

def loadImgRGBAsSize(path,size=(512,512)):
    img = Image.open(path)
    img = img.resize(size)
    if not img.mode == "RGB":
        img = img.convert("RGB")
    img = np.array(img).astype(np.uint8)
    return img

def predict_from_real_crop(opt,model,background_path,rain_layer_path,out_path,steps=50,use_lighten=True,use_blend=True):
    batch_size = 1
    first_key = model.first_stage_key
    cond_key = model.cond_stage_key
    patch_size = opt.hrig.data.params.validation.params.size


    background = loadImgRGB(background_path)
    rain_layer = loadImgRGBAsSize(rain_layer_path,size=(patch_size,patch_size))
    background = np.array(background).astype(np.float32)
    rain_layer = np.array(rain_layer).astype(np.float32)
    

    mask = rain_layer[...,0] /255.0
    mask[mask < 0.01] = 0
    mask[mask >= 0.01] = 1
    mask = mask.reshape(mask.shape[0],mask.shape[1],1)

    rain_layer = rain_layer / 127.5 - 1.0

    imgShape = np.array(background.shape[:2])
    blocks = np.ceil(imgShape/patch_size)
    fullSize = blocks * patch_size
    backgroundFull = np.zeros((int(fullSize[0]),int(fullSize[1]),3))
    backgroundFull[:int(imgShape[0]),:int(imgShape[1])] = background


    batch_input = {
        first_key: [],
        cond_key: [],
        "rain_layer": []
    }
    # 分成512x512的块
    for x in range(int(blocks[0])):
        for y in range(int(blocks[1])):
            background_patch = backgroundFull[x*patch_size:(x+1)*patch_size,y*patch_size:(y+1)*patch_size]
            masked_background = (1-mask)*background_patch

            background_patch = background_patch / 127.5 - 1.0
            masked_background = masked_background / 127.5 - 1.0

            batch_input[first_key].append(background_patch)
            batch_input[cond_key].append(masked_background)
            batch_input["rain_layer"].append(rain_layer)

    batch_input[first_key] = torch.tensor(np.array(batch_input[first_key]))
    batch_input[cond_key] = torch.tensor(np.array(batch_input[cond_key]))
    batch_input["rain_layer"] = torch.tensor(np.array(batch_input["rain_layer"]))

    # 按batch_size预测
    patch_num = int(blocks[0]*blocks[1])
    batch_num = int(np.ceil( patch_num / batch_size ))
    predicted_patchs = []
    for input_index in range(batch_num):
        b_input = {
            first_key: batch_input[first_key][input_index*batch_size:(input_index+1)*batch_size],
            cond_key: batch_input[cond_key][input_index*batch_size:(input_index+1)*batch_size],
            "rain_layer": batch_input["rain_layer"][input_index*batch_size:(input_index+1)*batch_size]
        }
        pred_size = len(b_input[first_key])
        x_samples_ddim = model.get_samples(b_input, pred_size, ddim_steps=steps)
        predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                        min=0.0, max=1.0)
        predicted_image = predicted_image.cpu().numpy().transpose(0,2,3,1)*255
        predicted_patchs.append(predicted_image)
    
    predicted_patchs = np.concatenate(predicted_patchs)
    patch_index = 0
    rainy_image = np.zeros(backgroundFull.shape)
    for x in range(int(blocks[0])):
        for y in range(int(blocks[1])):
            rainy_image_patch = predicted_patchs[patch_index]
            background_patch = backgroundFull[x*patch_size:(x+1)*patch_size,y*patch_size:(y+1)*patch_size]
            # 变亮后处理
            if use_lighten:
                rainy_image_patch = np.max((background_patch,rainy_image_patch),axis=0)
            # 与背景图像混合后处理
            if use_blend:
                rainy_image_patch = rainy_image_patch * mask + (1-mask) * background_patch
            rainy_image[x*patch_size:(x+1)*patch_size,y*patch_size:(y+1)*patch_size] = rainy_image_patch
            patch_index += 1
    rainy_image = rainy_image[:int(imgShape[0]),:int(imgShape[1])]

    Image.fromarray(rainy_image.astype(np.uint8)).save(out_path)


def predict_from_bg_mask(background_path,rain_layer_path,out_path,steps=50,use_lighten=True,use_blend=True):
    opt = OmegaConf.load(configFile)
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{opt.hrig.gpu}'

    logdir = opt.hrig.resume
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    

    # init and save configs
    configs_model = [OmegaConf.load(os.path.join(cfgdir,cfg)) for cfg in os.listdir(cfgdir)]
    configs_model = OmegaConf.merge(*configs_model)

    # model
    model = instantiate_from_config(configs_model.model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict(torch.load(os.path.join(ckptdir,opt.hrig.ckpt+".ckpt"),map_location=device)["state_dict"],
                          strict=False)
    print("Restored from "+os.path.join(ckptdir,opt.hrig.ckpt+".ckpt"))
    model = model.to(device)

    try:
        with torch.no_grad():
            predict_from_real_crop(opt,model,background_path,rain_layer_path,out_path,steps,use_lighten,use_blend)
        print("Generate Rainy Image Successfully!")
    except Exception as e:
        print(f"Generate Rainy Image Failed: {e}")

    del model
    torch.cuda.empty_cache()



if __name__=="__main__":
    background_path = "../Website/tasks_hrig/1c9b4387-fe85-4dfe-85b7-8eb25a9d1782/background.png"
    rain_layer_path = "../Website/tasks_hrig/1c9b4387-fe85-4dfe-85b7-8eb25a9d1782/rain.png"
    out_path = "../Website/tasks_hrig/1c9b4387-fe85-4dfe-85b7-8eb25a9d1782/output.png"
    predict_from_bg_mask(background_path,rain_layer_path,out_path)