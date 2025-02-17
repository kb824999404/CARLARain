import argparse, os, sys, datetime, glob, importlib, csv
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import ipdb
from einops import rearrange

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        type=str,
        const=True,
        default="last",
        nargs="?",
        help="name of ckpt file",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help="dataset root path",
    )
    parser.add_argument(
        "--data_json",
        type=str,
        help="json file of dataset",
    )
    parser.add_argument(
        "-p",
        "--pred",
        type=str,
        help="dataset root path",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=1,
        help="data batch size",
    )
    parser.add_argument(
        "--useResize",
        type=str,
        default="False",
        help="Input resized images",
    )
    parser.add_argument(
        "--pred_name",
        type=str,
        default=None,
        help="Predict result name",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Used GPU ID",
    )
    return parser


# 以masked_image为条件，用随机隐变量生成结果，结果与Ground Truth通过Mask混合
def predict_from_zero_blend(dataLoader):
    for batch_idx,batch in tqdm(enumerate(dataLoader)):
        image = (batch[model.cond_stage_key]+1.0)/2.0
        if "mask" in batch:
            mask = batch["mask"]
        else:
            mask = batch["mask_0"]
        batch_size = dataLoader.batch_size
        x_samples_ddim = model.get_samples(batch, batch_size, ddim_steps=opt.steps)
        predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                      min=0.0, max=1.0)
        

        predicted_image = predicted_image.cpu().numpy().transpose(0,2,3,1)
        image = image.cpu().numpy()
        mask = mask.unsqueeze(3).cpu().numpy()
        inpainted = mask*predicted_image+(1-mask)*image
        inpainted = inpainted*255


        for path,result in zip(batch['mask_path'],inpainted):
            path_new = os.path.split(path)[0][:-4] +"_pred.jpg"
            outpath = os.path.join(outdir,path_new)
            Image.fromarray(result.astype(np.uint8)).save(outpath)

# 以masked_image为条件，用随机隐变量生成结果
def predict_from_real_resize(dataLoader):
    for batch_idx,batch in tqdm(enumerate(dataLoader)):
        batch_size = dataLoader.batch_size
        x_samples_ddim = model.get_samples(batch, batch_size, ddim_steps=opt.steps)
        predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                        min=0.0, max=1.0)
        predicted_image = predicted_image.cpu().numpy().transpose(0,2,3,1)*255

        for index in range(batch_size):
            path = batch['background_path'][index]
            result = predicted_image[index]
            path_new = "{}_{}_{}_{}.jpg".format(batch["intensity"][index],
                                 batch["wind"][index],
                                 os.path.split(batch["background_path"][index])[-1][:-4],
                                 os.path.split(batch["rain_layer_path"][index])[-1][:-4])
            outpath = os.path.join(outdir,path_new)
            Image.fromarray(result.astype(np.uint8)).save(outpath)

def loadImgRGB(path):
    img = Image.open(path)
    if not img.mode == "RGB":
        img = img.convert("RGB")
    img = np.array(img).astype(np.uint8)
    return img

def predict_from_real_crop(dataLoader):
    for batch_idx,batch in tqdm(enumerate(dataLoader)):
        print("Predict Batch: {}/{}".format(batch_idx,len(dataLoader)))
        batch_size = dataLoader.batch_size
        first_key = model.first_stage_key
        cond_key = model.cond_stage_key
        patch_size = dataLoader.dataset.size
        for index in range(batch_size):
            path_new = "{}_{}_{}_{}.jpg".format(batch["intensity"][index],
                                 batch["wind"][index],
                                 os.path.split(batch["background_path"][index])[-1][:-4],
                                 os.path.split(batch["rain_layer_path"][index])[-1][:-4]
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
            fullSize = blocks * patch_size
            padding = fullSize - imgShape
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

                    # img_cond = (((masked_background+1.0)/2.0) * 255).astype(np.uint8)
                    # Image.fromarray(img_cond).save(os.path.join(outdir,"cond.jpg"))

                    # img_patch = (backgroundFull[x*patch_size:(x+1)*patch_size,y*patch_size:(y+1)*patch_size]).astype(np.uint8)
                    # Image.fromarray(img_patch).save(os.path.join(outdir,"patch.jpg"))

                    # img_backgroundFull = (backgroundFull).astype(np.uint8)
                    # Image.fromarray(img_backgroundFull).save(os.path.join(outdir,"backgroundFull.jpg"))

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
                x_samples_ddim = model.get_samples(b_input, pred_size, ddim_steps=opt.steps)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                                min=0.0, max=1.0)
                predicted_image = predicted_image.cpu().numpy().transpose(0,2,3,1)*255
                predicted_patchs.append(predicted_image)
            
            predicted_patchs = np.concatenate(predicted_patchs)
            patch_index = 0
            rainy_image = np.zeros(backgroundFull.shape)
            for x in range(int(blocks[0])):
                for y in range(int(blocks[1])):
                    rainy_image[x*patch_size:(x+1)*patch_size,y*patch_size:(y+1)*patch_size] = predicted_patchs[patch_index]
                    patch_index += 1
            rainy_image = rainy_image[:int(imgShape[0]),:int(imgShape[1])]
                    

            Image.fromarray(rainy_image.astype(np.uint8)).save(outpath)


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)
    

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    outdir = os.path.join(logdir,"results-real-"+opt.ckpt)
    if opt.pred_name:
        outdir = outdir + "-" + opt.pred_name

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    if opt.pred:
        config_pred = OmegaConf.load(os.path.join(opt.resume,"configs",opt.pred))
        config.data = config_pred.data

    # model
    model = instantiate_from_config(config.model)
    device = torch.device("cuda",opt.gpu) if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict(torch.load(os.path.join(ckptdir,opt.ckpt+".ckpt"),map_location=device)["state_dict"],
                          strict=False)
    print("Restored from "+os.path.join(ckptdir,opt.ckpt+".ckpt"))
    model = model.to(device)
    # data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    os.makedirs(outdir, exist_ok=True)


    with torch.no_grad():
        with model.ema_scope():
            valDataLoader = data._val_dataloader()
            if opt.useResize == "True":
                predict_from_real_resize(valDataLoader)
            else:
                predict_from_real_crop(valDataLoader)
