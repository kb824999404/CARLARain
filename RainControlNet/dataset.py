import json
import cv2
import numpy as np
import os
import random

from torch.utils.data import Dataset
from PIL import Image

class HRIGDataset(Dataset):
    def __init__(self,data_root,data_json,max_len=None,resolution=None):
        self.data_root = data_root
        with open(os.path.join(data_root,data_json), 'r') as f:
            self.data = json.load(f)
        if max_len:
            self.data = self.data[:max_len]
        self.data = [
            {
                "scene": l["scene"],
                "sequence": l["sequence"],
                "intensity": l["intensity"],
                "wind": l["wind"],
                "background_path": os.path.join(self.data_root,l["background"]),
                "depth_path": os.path.join(self.data_root,l["depth"]),
                "rain_layer_path": os.path.join(self.data_root,l["rain_layer"]),
                "rainy_depth_path": os.path.join(self.data_root,l["rainy_depth"]),
                "rainy_image_path": os.path.join(self.data_root,l["rainy_image"]),
            }
            for l in self.data
        ]

        self.resolution = resolution

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        background = np.array(Image.open(item['background_path']).convert("RGB"))
        rainy_image = np.array(Image.open(item['rainy_image_path']).convert("RGB"))
        rain_layer = np.array(Image.open(item['rain_layer_path']).convert("RGBA"))

        # Random clip to the target resolution
        if self.resolution:
            clipX = np.random.randint(0, background.shape[0] - self.resolution)
            clipY = np.random.randint(0, background.shape[1] - self.resolution)
            background = background[clipX:clipX+self.resolution,clipY:clipY+self.resolution]
            rainy_image = rainy_image[clipX:clipX+self.resolution,clipY:clipY+self.resolution]
            rain_layer = rain_layer[clipX:clipX+self.resolution,clipY:clipY+self.resolution]



        #import ipdb
        #ipdb.set_trace()
        # Rain Mask
        mask = rain_layer[...,-1].astype(np.float32) /255.0
        mask[mask < 0.01] = 0
        mask[mask >= 0.01] = 1

        mask = mask[...,None]

        masked_background = (1-mask)*background

        # Normalize source images to [0, 1].
        masked_background = masked_background.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        rainy_image = (rainy_image.astype(np.float32) / 127.5) - 1.0

        prompt = "rainy scene"

        return dict(jpg=rainy_image, txt=prompt, hint=masked_background)



class CRIGDataset(Dataset):
    def __init__(self,data_root,data_json,max_len=None,resolution=None,hint_size=64,
                 max_intensity = 100):
        self.data_root = data_root
        with open(os.path.join(data_root,data_json), 'r') as f:
            self.data = json.load(f)
        if max_len:
            self.data = self.data[:max_len]

        self.data = [
            {
                "intensity": l["intensity"],
                "wind": l["wind"],
                "rainy_image_path": os.path.join(self.data_root,l["rainy_image"]),
                "rain_mask_path": os.path.join(self.data_root,l["rain_mask"]),
            }
            for l in self.data if l["intensity"] <= max_intensity
        ]

        self.resolution = resolution
        self.hint_size = hint_size
        self.max_intensity = max_intensity

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        rainy_image = np.array(Image.open(item['rainy_image_path']).convert("RGB"))

        
        # Random clip to the target resolution
        if self.resolution:
            clipX = np.random.randint(0, rainy_image.shape[0] - self.resolution)
            clipY = np.random.randint(0, rainy_image.shape[1] - self.resolution)
            rainy_image = rainy_image[clipX:clipX+self.resolution,clipY:clipY+self.resolution]


        startX = np.random.randint(0, rainy_image.shape[0] - self.hint_size)
        startY = np.random.randint(0, rainy_image.shape[1] - self.hint_size)


        mask = np.zeros((rainy_image.shape[0],rainy_image.shape[1],1))
        mask[startX:startX+self.hint_size,startY:startY+self.hint_size] = 1.0

        hint_image = rainy_image.copy()
        hint_image = mask * rainy_image

        # Normalize source images to [0, 1].
        hint_image = hint_image.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        rainy_image = (rainy_image.astype(np.float32) / 127.5) - 1.0

        label = np.array([ 
                item["intensity"]/self.max_intensity,
                (item["wind"]+1.0)/2.0,
            ]).astype(np.float32)

        prompt = "rain"

        return dict(jpg=rainy_image, txt=prompt, hint=hint_image, label=label)



class CRIGDatasetFolder(Dataset):
    def __init__(self,data_root,max_len=None,resolution=512):
        self.data_root = data_root
        IMG_EXTENSIONS = [ "jpg","png","jpeg"  ]
        self.data = [
            file for file in os.listdir(data_root) if os.path.splitext(file)[1][1:] in IMG_EXTENSIONS
        ]

        if max_len:
            self.data = self.data[:max_len]

        self.resolution = resolution


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        rain_lr = np.array(Image.open(os.path.join(self.data_root,item)).convert("RGB"))

        rain_padding = np.zeros((self.resolution,self.resolution,3))

        startX = np.random.randint(0, self.resolution - rain_lr.shape[0])
        startY = np.random.randint(0, self.resolution - rain_lr.shape[1])

        rain_padding[startX:startX+rain_lr.shape[0],startY:startY+rain_lr.shape[1]] = rain_lr
        rain_padding_copy = rain_padding.copy()

        # Normalize source images to [0, 1].
        rain_padding = rain_padding.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        rain_padding_copy = (rain_padding_copy.astype(np.float32) / 127.5) - 1.0

        prompt = "rain"

        return dict(jpg=rain_padding_copy, txt=prompt, hint=rain_padding, fileName=item)
    


class CRIGDatasetFolderWithLabel(Dataset):
    def __init__(self,data_root,data_json,max_len=None,resolution=512):
        self.data_root = data_root
        with open(os.path.join(data_root,data_json), 'r') as f:
            self.data = json.load(f)

        if max_len:
            self.data = self.data[:max_len]


        self.resolution = resolution


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        rain_lr = np.array(Image.open(os.path.join(self.data_root,item['img'])).convert("RGB"))

        rain_padding = np.zeros((self.resolution,self.resolution,3))

        startX = np.random.randint(0, self.resolution - rain_lr.shape[0])
        startY = np.random.randint(0, self.resolution - rain_lr.shape[1])

        rain_padding[startX:startX+rain_lr.shape[0],startY:startY+rain_lr.shape[1]] = rain_lr
        rain_padding_copy = rain_padding.copy()

        # Normalize source images to [0, 1].
        rain_padding = rain_padding.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        rain_padding_copy = (rain_padding_copy.astype(np.float32) / 127.5) - 1.0

        # [-1,1] -> [0,1]
        label = item['label']
        label = (np.array(item['label']).astype(np.float32) + 1.0 ) * 0.5

        prompt = "rain"

        return dict(jpg=rain_padding_copy, txt=prompt, hint=rain_padding, label=label, fileName=item['img'])