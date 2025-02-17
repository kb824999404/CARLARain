import argparse, os, sys, datetime, glob, importlib, csv
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from torchvision.transforms import Resize
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
        "-bs",
        "--batch_size",
        type=int,
        default=1,
        help="data batch size",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Used GPU ID",
    )
    parser.add_argument(
        "-g",
        "--guiding",
        type=str,
        default="False",
        help="Use GDM",
    )
    return parser

# 以masked_image为条件，用随机隐变量生成结果
def predict_from_zero(dataLoader):
    for batch_idx,batch in tqdm(enumerate(dataLoader)):
        batch_size = dataLoader.batch_size
        x_samples_ddim = model.get_samples(batch, batch_size, ddim_steps=opt.steps)
        predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                        min=0.0, max=1.0)
        predicted_image = predicted_image.cpu().numpy().transpose(0,2,3,1)*255

        for path,result in zip(batch['mask_path'],predicted_image):
            path_new = os.path.split(path)[-1][:-4] +"_pred.jpg"
            outpath = os.path.join(outdir,path_new)
            Image.fromarray(result.astype(np.uint8)).save(outpath)

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
        inpainted = (1-mask)*predicted_image+mask*image
        inpainted = inpainted*255


        for path,result in zip(batch['mask_path'],inpainted):
            path_new = os.path.split(path)[-1][:-4] +"_pred.jpg"
            outpath = os.path.join(outdir,path_new)
            Image.fromarray(result.astype(np.uint8)).save(outpath)

# 将图像裁剪为模型输入尺寸输入，得到的输出再合并
def predict_parts_from_zero_blend(dataLoader):
    for batch_idx,batch in enumerate(dataLoader):
        print("Predict Batch: {}/{}".format(batch_idx,len(dataLoader)))
        batch_size = dataLoader.batch_size
        first_key = model.first_stage_key
        cond_key = model.cond_stage_key
        if "mask" in batch:
            mask_key = "mask"
        else:
            mask_key = "mask_0"
        patch_size = dataLoader.dataset.size
        img_size = batch[first_key].shape[1:3]
        countX, countY = int(np.ceil(img_size[0] / patch_size)), int(np.ceil(img_size[1] / patch_size))
        predicted_image = np.zeros(shape=batch[first_key].shape)
        for x in range(countX):
            for y in range(countY):
                batch_patch = {}
                batch_patch[first_key] = batch[first_key][:,x*patch_size:(x+1)*patch_size,y*patch_size:(y+1)*patch_size]
                batch_patch[cond_key] = batch[cond_key][:,x*patch_size:(x+1)*patch_size,y*patch_size:(y+1)*patch_size]
                # img_cond = (((batch[cond_key][0]+1.0)/2.0).cpu().numpy()* 255).astype(np.uint8)
                # Image.fromarray(img_cond).save(os.path.join(outdir,"cond.jpg"))
                # patch_cond = (((batch_patch[cond_key][0]+1.0)/2.0).cpu().numpy()* 255).astype(np.uint8)
                # Image.fromarray(patch_cond).save(os.path.join(outdir,"cond_patch_{}_{}.jpg".format(x,y)))
                # patch_gt = (((batch_patch[first_key][0]+1.0)/2.0).cpu().numpy()* 255).astype(np.uint8)
                # Image.fromarray(patch_gt).save(os.path.join(outdir,"gt_patch_{}_{}.jpg".format(x,y)))
                x_samples_ddim = model.get_samples(batch_patch, batch_size, ddim_steps=opt.steps)
                predicted_patch = torch.clamp((x_samples_ddim+1.0)/2.0,min=0.0, max=1.0)
                predicted_patch = predicted_patch.cpu().numpy().transpose(0,2,3,1)
                predicted_image[:,x*patch_size:(x+1)*patch_size,y*patch_size:(y+1)*patch_size] = predicted_patch

        image = ((batch[cond_key]+1.0)/2.0).cpu().numpy()
        mask = batch[mask_key].unsqueeze(3).cpu().numpy()

        inpainted = mask*predicted_image+(1-mask)*image
        inpainted = inpainted*255
        predicted_image = predicted_image*255
        for index in range(len(inpainted)):
            path_new = "{}_{}_{}_{}_{}_pred.jpg".format(
                                                        batch["scene"][index],
                                                        batch["sequence"][index],
                                                        batch["intensity"][index],
                                                        batch["wind"][index],
                                                        os.path.split(batch["background_path"][index])[-1][:-4]
                                                    )
            outpath = os.path.join(outdir,path_new)
            Image.fromarray(inpainted[index].astype(np.uint8)).save(outpath)

            outpath_noblend = os.path.join(outdir_noblend,path_new)
            Image.fromarray(predicted_image[index].astype(np.uint8)).save(outpath_noblend)

# 将图像裁剪为模型输入尺寸输入，得到的输出再合并
def predict_parts_from_zero_blend_guiding(dataLoader):
    for batch_idx,batch in enumerate(dataLoader):
        print("Predict Batch: {}/{}".format(batch_idx,len(dataLoader)))
        batch_size = dataLoader.batch_size
        first_key_0 = model.first_stage_key
        cond_key_0 = model.cond_stage_key
        first_key_1 = model.gdm.first_stage_key
        cond_key_1 = model.gdm.cond_stage_key
        if "mask" in batch:
            mask_key = "mask"
        else:
            mask_key = "mask_0"
        patch_size = dataLoader.dataset.size[0]
        patch_size_gdm = dataLoader.dataset.size[1]
        img_size = batch[first_key_0].shape[1:3]
        countX, countY = int(np.ceil(img_size[0] / patch_size)), int(np.ceil(img_size[1] / patch_size))
        predicted_image = np.zeros(shape=batch[first_key_0].shape)
        for x in range(countX):
            for y in range(countY):
                batch_patch = {}
                batch_patch[cond_key_0] = batch[cond_key_0][:,x*patch_size:(x+1)*patch_size,y*patch_size:(y+1)*patch_size]
                batch_patch[first_key_0] = batch_patch[cond_key_0]
                batch_patch[cond_key_1] = batch[cond_key_1][:,x*patch_size_gdm:(x+1)*patch_size_gdm,y*patch_size_gdm:(y+1)*patch_size_gdm]
                batch_patch[first_key_1] = batch_patch[cond_key_1]
                x_samples_ddim = model.get_samples(batch_patch, batch_size, ddim_steps=opt.steps)
                predicted_patch = torch.clamp((x_samples_ddim+1.0)/2.0,min=0.0, max=1.0)
                predicted_patch = predicted_patch.cpu().numpy().transpose(0,2,3,1)
                predicted_image[:,x*patch_size:(x+1)*patch_size,y*patch_size:(y+1)*patch_size] = predicted_patch

        image = ((batch[cond_key_0]+1.0)/2.0).cpu().numpy()
        mask = batch[mask_key].unsqueeze(3).cpu().numpy()

        inpainted = mask*predicted_image+(1-mask)*image
        inpainted = inpainted*255
        predicted_image = predicted_image*255
        for index in range(len(inpainted)):
            path_new = "{}_{}_{}_{}_{}_pred.jpg".format(
                                                        batch["scene"][index],
                                                        batch["sequence"][index],
                                                        batch["intensity"][index],
                                                        batch["wind"][index],
                                                        os.path.split(batch["background_path"][index])[-1][:-4]
                                                    )
            outpath = os.path.join(outdir,path_new)
            Image.fromarray(inpainted[index].astype(np.uint8)).save(outpath)

            outpath_noblend = os.path.join(outdir_noblend,path_new)
            Image.fromarray(predicted_image[index].astype(np.uint8)).save(outpath_noblend)

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
    outdir = os.path.join(logdir,"results-"+opt.ckpt)
    outdir_noblend = os.path.join(logdir,"results_noblend-"+opt.ckpt)

    # init and save configs
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    # model
    model = instantiate_from_config(config.model)
    device = torch.device("cuda",opt.gpu) if torch.cuda.is_available() else torch.device("cpu")
    model.load_state_dict(torch.load(os.path.join(ckptdir,opt.ckpt+".ckpt"),map_location=device)["state_dict"],
                          strict=False)
    print("Restored from "+os.path.join(ckptdir,opt.ckpt+".ckpt"))
    model = model.to(device)
    # data
    config.data.params.train.params.fullSize = True
    config.data.params.validation.params.fullSize = True
    config.data.params.batch_size = opt.batch_size
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_noblend, exist_ok=True)

    with torch.no_grad():
        with model.ema_scope():
            trainDataLoader = data._train_dataloader()
            testDataLoader = data._val_dataloader()
            
            if opt.guiding == "False":
                predict_parts_from_zero_blend(testDataLoader)
            else:
                predict_parts_from_zero_blend_guiding(testDataLoader)