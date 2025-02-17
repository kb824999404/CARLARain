import os
import json
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import color
import ipdb
from random import choice
from natsort import natsorted

class RainBlenderReal(Dataset):
    def __init__(self,
                 json_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 readImg=True
                 ):
        self.data_paths = json_file
        self.data_root = data_root
        with open(self.data_paths,"r") as f:
            self.image_paths = json.load(f)
        self._length = len(self.image_paths)


        self.labels = [
            {
                'wind': l['wind'],
                'intensity': l['intensity'],
                "background_path": os.path.join(self.data_root,l["background"]),
                "rain_mask_path": os.path.join(self.data_root,l["rain_mask"]),
                "rain_layer_path": os.path.join(self.data_root,l["rainy_image"]),
            }
            for l in self.image_paths
        ]

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        self.readImg = readImg

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = self.labels[i].copy()

        if not self.readImg:
            return example
        
        background = loadImgRGB(example["background_path"],False)
        rain_layer = loadImgRGB(example["rain_layer_path"],False)


        background = background.resize((self.size, self.size), resample=self.interpolation)
        rain_layer = rain_layer.resize((self.size, self.size), resample=self.interpolation)

        # Background
        example["raw_background"] = np.array(background)
        example["background"] = np.array(background).astype(np.float32) / 127.5 - 1.0


        # Rain Layer
        example["raw_rain_layer"] = np.array(rain_layer)
        rain_layer = np.array(rain_layer).astype(np.float32)
        example["rain_layer"] = rain_layer / 127.5 - 1.0
        example["rainy_image"] = example["rain_layer"]
        
        # Rain Mask
        mask = rain_layer[...,0] /255.0
        mask[mask < 0.01] = 0
        mask[mask >= 0.01] = 1

        example["mask"] = mask
        mask = mask.reshape(mask.shape[0],mask.shape[1],1)

        masked_background = (1-mask)*example["raw_background"]
        example["masked_background"] = masked_background / 127.5 - 1.0


        return example
    
class RainCARLA(RainBlenderReal):
    def __init__(self,
                 data_root,
                 background_path,
                 rain_path,
                 size=None,
                 interpolation="bicubic",
                 readImg=True
                 ):
        IMAGE_EXTENT = [ "jpg","png","jpeg"  ]
        self.data_root = data_root
        self.background_root = os.path.join(data_root,background_path)
        self.rain_root = os.path.join(data_root,rain_path)
        self.image_paths = [  os.path.join(self.background_root,f) for f in natsorted(os.listdir(self.background_root)) if f.split(".")[-1] in IMAGE_EXTENT  ]
        self.rain_paths = [  os.path.join(self.rain_root,f) for f in natsorted(os.listdir(self.rain_root)) if f.split(".")[-1] in IMAGE_EXTENT ]
        self._length = len(self.image_paths)

        self.labels = []
        for bg in self.image_paths:
            self.labels.append({
                "background_path": bg,
                "rain_layer_path": choice(self.rain_paths)
            })


        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        self.readImg = readImg


class RainBlenderRealList(RainBlenderReal):
    def __getitem__(self, i):
        example = self.labels[i].copy()
        background = loadImgRGB(example["background_path"], not self.randomClip)
        rainy_image = loadImgRGB(example["rainy_image_path"], not self.randomClip)
        rain_layer = loadImgRGBA(example["rain_layer_path"], not self.randomClip)
        if self.randomClip:
            clipX = np.random.randint(0, background.size[0] - self.size[0])
            clipY = np.random.randint(0, background.size[1] - self.size[0])
            background = background.crop((clipX,clipY,clipX+self.size[0],clipY+self.size[0]))
            rainy_image = rainy_image.crop((clipX,clipY,clipX+self.size[0],clipY+self.size[0]))
            rain_layer = rain_layer.crop((clipX,clipY,clipX+self.size[0],clipY+self.size[0]))
        else:
            background = background.resize((self.size, self.size), resample=self.interpolation)
            rainy_image = rainy_image.resize((self.size, self.size), resample=self.interpolation)
            rain_layer = rain_layer.resize((self.size, self.size), resample=self.interpolation)

        if self.useDepth:
            depth = loadImgGray(example["depth_path"], not self.randomClip)
            rainy_depth = loadImgGray(example["rainy_depth_path"], not self.randomClip)
            if self.randomClip:
                depth = depth.crop((clipX,clipY,clipX+self.size[0],clipY+self.size[0]))
                rainy_depth = rainy_depth.crop((clipX,clipY,clipX+self.size[0],clipY+self.size[0]))
                
        for size_idx, size_hw in enumerate(self.size):    
            # Background
            background_reszie = background.resize((size_hw, size_hw), resample=self.interpolation)
            example["raw_background_%d" % size_idx] = np.array(background_reszie)
            example["background_%d" % size_idx] = np.array(background_reszie).astype(np.float32) / 127.5 - 1.0

            # Rainy Image
            rainy_image_reszie = rainy_image.resize((size_hw, size_hw), resample=self.interpolation)
            example["raw_rainy_image_%d" % size_idx] = np.array(rainy_image_reszie)
            example["rainy_image_%d" % size_idx] = np.array(rainy_image_reszie).astype(np.float32) / 127.5 - 1.0

            # Rain Layer
            rain_layer_reszie = rain_layer.resize((size_hw, size_hw), resample=self.interpolation)
            example["raw_rain_layer_%d" % size_idx] = np.array(rain_layer_reszie)
            rain_layer_reszie = np.array(rain_layer_reszie).astype(np.float32)
            example["rain_layer_%d" % size_idx] = rain_layer_reszie[...,:-1] / 127.5 - 1.0

            # Rain Mask
            mask = rain_layer_reszie[...,-1] /255.0
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1

            example["mask_%d" % size_idx] = mask
            mask = mask.reshape(size_hw, size_hw,1)

            masked_background = (1-mask)*background_reszie
            example["masked_background_%d"% size_idx] = masked_background / 127.5 - 1.0

            if self.useDepth:
                # Depth
                depth_reszie = depth.resize((size_hw, size_hw), resample=self.interpolation)
                example["raw_depth_%d" % size_idx] = np.array(depth_reszie)
                example["depth_%d" % size_idx] = np.array(depth_reszie).astype(np.float32) / 127.5 - 1.0
                
                # Rainy Depth
                rainy_depth_reszie = rainy_depth.resize((size_hw, size_hw), resample=self.interpolation)
                example["raw_rainy_depth_%d" % size_idx] = np.array(rainy_depth_reszie)
                example["rainy_depth_%d" % size_idx] = np.array(rainy_depth_reszie).astype(np.float32) / 127.5 - 1.0

        return example
    

def loadImgRGB(path,clip):
    img = Image.open(path)
    if not img.mode == "RGB":
        img = img.convert("RGB")

    if clip:
        img = np.array(img).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]
        img = Image.fromarray(img)

    return img

def loadImgRGBA(path,clip):
    img = Image.open(path)
    if not img.mode == "RGBA":
        img = img.convert("RGBA")

    if clip:
        img = np.array(img).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]
        img = Image.fromarray(img)

    return img


def loadImgGray(path,clip):
    img = Image.open(path)
    if not img.mode == "L":
        img = img.convert("L")

    if clip:
        img = np.array(img).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]
        img = Image.fromarray(img)

    return img