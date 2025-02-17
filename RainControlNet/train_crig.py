# 训练Controlnet Seg2Image模型
from share import *

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
try:
    from pytorch_lightning.strategies import DDPStrategy
except:
    print("Error import: No DDPStrategy in pytorch_lightning.strategies!")

from torch.utils.data import DataLoader
from dataset import CRIGDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

import os
import argparse
import datetime
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Rain ControlNet."
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
        default="/home/zhoukaibin/data7/dataset/BlenderRain",
        help="Dataset root.",
    )
    parser.add_argument(
        "--data_json",
        type=str,
        default="trainset.json",
        help="Dataset json file.",
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
        "--hint_size",
        type=int,
        default=64,
        help="The size of hint image.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="The resolution of original image.",
    )
    parser.add_argument(
        "--logger_freq",
        type=int,
        default=300,
        help="The frequency of logger.",
    )
    parser.add_argument(
        "--ckpt_save_freq",
        type=int,
        default=5000,
        help="The frequency of saving checkpoints(steps).",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        type=float,
        default=1e-5,
        help="The learning rate.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=20000,
        help="The max training steps.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="The max len of dataset.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="[0]",
        help="The gpu devices.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs",
        help="The save dir of logs.",
    )
    parser.add_argument(
        "--gradient_accumulate",
        type=int,
        default=1,
        help="The steps of accumulated gradients.",
    )

    args = parser.parse_args()
    return args

@rank_zero_only
def save_args(args):
    os.makedirs(args.log_dir,exist_ok=True)
    os.makedirs(os.path.join(args.log_dir,args.task),exist_ok=True)
    with open(os.path.join(args.log_dir,args.task,"args.json"),"w") as f:
        # Save configs as file
        json.dump(vars(args),f,indent=4)

if __name__=="__main__":
    # Configs
    args = parse_args()
    args.task += datetime.datetime.now().strftime("-%Y-%m-%d-T%H-%M-%S")
    save_args(args)

    sd_locked = True
    only_mid_control = False

    # Create Model
    model = create_model(args.model_config).cpu()
    resume_path = './models/control_sd15_ini_rain.ckpt'
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))

    model.learning_rate = args.learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Load Data
    dataset = CRIGDataset(args.data_root,args.data_json,max_len=args.max_len,resolution=args.resolution,hint_size=args.hint_size)
    print("Dataset Size:",len(dataset))
    dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    # Create Trainer
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.task)
    imageLogger = ImageLogger(task_name=args.task,batch_frequency=args.logger_freq)
    #checkpoint_callback = ModelCheckpoint(every_n_train_steps=args.ckpt_save_freq,save_top_k=-1)
    checkpoint_callback = ModelCheckpoint(every_n_train_steps=args.ckpt_save_freq)
    devices = eval(args.gpus)
    if len(devices) > 1:
        strategy = "ddp"
        #strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = None
    if args.resume:
        trainer = pl.Trainer(accelerator="gpu", strategy=strategy, devices = devices, 
                             max_steps=args.max_steps, precision=16,
                             logger=logger, callbacks=[imageLogger,checkpoint_callback], 
                             accumulate_grad_batches=args.gradient_accumulate, gradient_clip_algorithm = "norm", gradient_clip_val = 10,
                             resume_from_checkpoint=args.resume)
    else:
        trainer = pl.Trainer(accelerator="gpu", strategy=strategy, devices = devices, 
                             max_steps=args.max_steps, precision=16,
                             logger=logger, callbacks=[imageLogger,checkpoint_callback], 
                             accumulate_grad_batches=args.gradient_accumulate, gradient_clip_algorithm = "norm", gradient_clip_val = 10)
    # Train
    trainer.fit(model, dataloader)

