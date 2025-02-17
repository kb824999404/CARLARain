import argparse, os, sys, datetime, glob
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from main import instantiate_from_config
import ipdb
import yaml

def get_args():
    parser = argparse.ArgumentParser(description='HRIG Predict')
    parser.add_argument('-c', '--config',
                        type=str,
                        default=None,
                        help='The config file')    
    
    args = parser.parse_args()

    if args.config:
        assert os.path.exists(args.config), ("The config file is missing.", args.config)
        cfg = OmegaConf.load(args.config)
        return cfg


def loadImgRGB(path):
    img = Image.open(path)
    if not img.mode == "RGB":
        img = img.convert("RGB")
    img = np.array(img).astype(np.uint8)
    return img

def predict_from_real_crop(dataLoader):
    for batch_idx,batch in enumerate(dataLoader):
        print("Predict Batch: {}/{}".format(batch_idx,len(dataLoader)))
        batch_size = dataLoader.batch_size
        first_key = model.first_stage_key
        cond_key = model.cond_stage_key
        patch_size = dataLoader.dataset.size
        for index in range(batch_size):
            path_new = "{}.jpg".format(
                                 os.path.split(batch["background_path"][index])[-1][:-4]
                                 )
            outpath = os.path.join(outdir,path_new)
            if os.path.exists(outpath):
                continue

            background = loadImgRGB(batch["background_path"][index])
            rain_layer = loadImgRGB(batch["rain_layer_path"][index])
            background = np.array(background).astype(np.float32)
            rain_layer = np.array(rain_layer).astype(np.float32)

            mask = rain_layer[...,0] /255.0
            mask[mask < 0.01] = 0
            mask[mask >= 0.01] = 1
            mask = mask.reshape(mask.shape[0],mask.shape[1],1)

            rain_layer = rain_layer / 127.5 - 1.0

            imgShape = np.array(background.shape[:2])
            blocks = np.ceil(imgShape/patch_size)
            fullSize = blocks * patch_size             # patch_size的整数倍
            overlap = opt.hrig.overlap                 # 重叠区域像素
            step = patch_size - overlap
            backgroundFull = np.zeros((int(fullSize[0]),int(fullSize[1]),3))
            backgroundFull[:int(imgShape[0]),:int(imgShape[1])] = background
            maskFull = np.zeros((int(fullSize[0]),int(fullSize[1]),1))
            rainlayerFull = np.zeros((int(fullSize[0]),int(fullSize[1]),3))
            for x in range(0, backgroundFull.shape[0],patch_size):
                for y in range(0, backgroundFull.shape[1],patch_size):
                    maskFull[x:x+patch_size,y:y+patch_size] = mask
                    rainlayerFull[x:x+patch_size,y:y+patch_size] = rain_layer

            batch_input = {
                first_key: [],
                cond_key: [],
                "rain_layer": []
            }
            # 分成512x512的块
            for x in range(0, backgroundFull.shape[0], step):
                for y in range(0, backgroundFull.shape[1], step):
                    background_patch = np.zeros((int(patch_size),int(patch_size),3))
                    background_patch[:backgroundFull.shape[0]-x,:backgroundFull.shape[1]-y] = backgroundFull[x:x+patch_size,y:y+patch_size]

                    mask_patch = np.zeros((int(patch_size),int(patch_size),1))
                    mask_patch[:backgroundFull.shape[0]-x,:backgroundFull.shape[1]-y] = maskFull[x:x+patch_size,y:y+patch_size]

                    masked_background = (1-mask_patch)*background_patch

                    rainylayer_patch = np.zeros((int(patch_size),int(patch_size),3))
                    rainylayer_patch[:backgroundFull.shape[0]-x,:backgroundFull.shape[1]-y] = rainlayerFull[x:x+patch_size,y:y+patch_size]

                    background_patch = background_patch / 127.5 - 1.0
                    masked_background = masked_background / 127.5 - 1.0

                    batch_input[first_key].append(background_patch)
                    batch_input[cond_key].append(masked_background)
                    batch_input["rain_layer"].append(rainylayer_patch)


            batch_input[first_key] = torch.tensor(np.array(batch_input[first_key]))
            batch_input[cond_key] = torch.tensor(np.array(batch_input[cond_key]))
            batch_input["rain_layer"] = torch.tensor(np.array(batch_input["rain_layer"]))

            # 按batch_size预测
            patch_num = len(batch_input[first_key])
            batch_num = int(np.ceil( patch_num / batch_size ))
            predicted_patchs = []
            for input_index in range(batch_num):
                b_input = {
                    first_key: batch_input[first_key][input_index*batch_size:(input_index+1)*batch_size],
                    cond_key: batch_input[cond_key][input_index*batch_size:(input_index+1)*batch_size],
                    "rain_layer": batch_input["rain_layer"][input_index*batch_size:(input_index+1)*batch_size]
                }
                pred_size = len(b_input[first_key])
                x_samples_ddim = model.get_samples(b_input, pred_size, ddim_steps=opt.hrig.steps)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                                min=0.0, max=1.0)
                predicted_image = predicted_image.cpu().numpy().transpose(0,2,3,1)*255
                predicted_patchs.append(predicted_image)
            
            predicted_patchs = np.concatenate(predicted_patchs)
            patch_index = 0
            rainy_image = np.zeros(backgroundFull.shape)
            for x in range(0, backgroundFull.shape[0], step):
                for y in range(0, backgroundFull.shape[1], step):
                    rainy_image_patch = predicted_patchs[patch_index][:backgroundFull.shape[0]-x,:backgroundFull.shape[1]-y]
                    # 与背景图像混合后处理
                    if opt.hrig.use_lighten:
                        background_patch = backgroundFull[x:x+patch_size,y:y+patch_size]
                        rainy_image_patch = np.max((background_patch,rainy_image_patch),axis=0)
                        mask_patch = maskFull[x:x+patch_size,y:y+patch_size]
                        rainy_image_patch = rainy_image_patch * mask_patch + (1-mask_patch) * background_patch
                    #rainy_image[x:x+patch_size,y:y+patch_size] = \
                    #    0.5 * rainy_image[x:x+patch_size,y:y+patch_size] \
                    #    + 0.5 * rainy_image_patch
                    #rainy_image[x:x+patch_size,y:y+patch_size] = np.max((rainy_image[x:x+patch_size,y:y+patch_size],
                    #                                                     rainy_image_patch),axis=0)
                    if x > 0:
                        top_patch = rainy_image[x-overlap:x,y:y+patch_size]
                        rainy_image_patch[:overlap,:] = 0.5 * top_patch + 0.5 * rainy_image_patch[:overlap,:]
                    if y > 0:
                        left_patch = rainy_image[x:x+patch_size,y-overlap:y]
                        rainy_image_patch[:,:overlap] = 0.5 * left_patch + 0.5 * rainy_image_patch[:,:overlap]
                    rainy_image[x:x+patch_size,y:y+patch_size] = rainy_image_patch
                    patch_index += 1
            rainy_image = rainy_image[:int(imgShape[0]),:int(imgShape[1])]
                    

            Image.fromarray(rainy_image.astype(np.uint8)).save(outpath)


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    opt = get_args()

    dataRoot = opt.scene.dataRoot
    seqName = opt.scene.name
    seqRoot = os.path.join(dataRoot,seqName)

    logdir = opt.hrig.resume
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    outdir = os.path.join(seqRoot,"rainy_overlap")

    # init and save configs
    configs_model = [OmegaConf.load(os.path.join(cfgdir,cfg)) for cfg in os.listdir(cfgdir)]
    configs_model = OmegaConf.merge(*configs_model)


    # model
    model = instantiate_from_config(configs_model.model)
    device = torch.device("cuda",opt.hrig.gpu) if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict(torch.load(os.path.join(ckptdir,opt.hrig.ckpt+".ckpt"),map_location=device)["state_dict"],
                          strict=False)
    print("Restored from "+os.path.join(ckptdir,opt.hrig.ckpt+".ckpt"))
    model = model.to(device)
    data_param = opt.hrig.data.params.validation.params
    data_param.data_root = os.path.join(dataRoot,data_param.data_root)
    data_param.background_path = os.path.join(data_param.data_root,data_param.background_path)
    data_param.rain_path = os.path.join(data_param.data_root,data_param.rain_path)

    # data
    data = instantiate_from_config(opt.hrig.data)
    data.prepare_data()
    data.setup()
    os.makedirs(outdir, exist_ok=True)

    with torch.no_grad():
        with model.ema_scope():
            valDataLoader = data._val_dataloader()
            predict_from_real_crop(valDataLoader)

