import os
import json
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import color
import ipdb

class RainBlender(Dataset):
    def __init__(self,
                 json_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 fullSize=False,
                 randomClip=True,
                 useDepth=False
                 ):
        self.data_paths = json_file
        self.data_root = data_root
        with open(self.data_paths,"r") as f:
            self.image_paths = json.load(f)
        self._length = len(self.image_paths)
        self.labels = [
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
            for l in self.image_paths
        ]


        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        self.fullSize = fullSize
        self.randomClip = randomClip
        self.useDepth = useDepth

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = self.labels[i].copy()
        
        background = loadImgRGB(example["background_path"],not self.fullSize and not self.randomClip)
        rainy_image = loadImgRGB(example["rainy_image_path"],not self.fullSize and not self.randomClip)
        rain_layer = loadImgRGBA(example["rain_layer_path"],not self.fullSize and not self.randomClip)

        if not self.fullSize:
            if self.randomClip:
                clipX = np.random.randint(0, background.size[0] - self.size)
                clipY = np.random.randint(0, background.size[1] - self.size)
                background = background.crop((clipX,clipY,clipX+self.size,clipY+self.size))
                rainy_image = rainy_image.crop((clipX,clipY,clipX+self.size,clipY+self.size))
                rain_layer = rain_layer.crop((clipX,clipY,clipX+self.size,clipY+self.size))
            else:
                background = background.resize((self.size, self.size), resample=self.interpolation)
                rainy_image = rainy_image.resize((self.size, self.size), resample=self.interpolation)
                rain_layer = rain_layer.resize((self.size, self.size), resample=self.interpolation)

        # Background
        example["raw_background"] = np.array(background)
        example["background"] = np.array(background).astype(np.float32) / 127.5 - 1.0

        # Rainy Image
        example["raw_rainy_image"] = np.array(rainy_image)
        example["rainy_image"] = np.array(rainy_image).astype(np.float32) / 127.5 - 1.0

        # Rain Layer
        example["raw_rain_layer"] = np.array(rain_layer)
        rain_layer = np.array(rain_layer).astype(np.float32)
        example["rain_layer"] = rain_layer[...,:-1] / 127.5 - 1.0
        
        # Rain Mask
        mask = rain_layer[...,-1] /255.0
        mask[mask < 0.01] = 0
        mask[mask >= 0.01] = 1

        example["mask"] = mask
        mask = mask.reshape(mask.shape[0],mask.shape[1],1)

        masked_background = (1-mask)*example["raw_background"]
        example["masked_background"] = masked_background / 127.5 - 1.0

        if self.useDepth:
            # Depth
            depth = loadImgGray(example["depth_path"],not self.fullSize and not self.randomClip)
            if not self.fullSize:
                if self.randomClip:
                    depth = depth.crop((clipX,clipY,clipX+self.size,clipY+self.size))
                else:
                    depth = depth.resize((self.size, self.size), resample=self.interpolation)
            example["raw_depth"] = np.array(depth)
            example["depth"] = np.array(depth).astype(np.float32) / 127.5 - 1.0

            # Rainy Depth
            rainy_depth = loadImgGray(example["rainy_depth_path"], not self.fullSize and not self.randomClip)
            if not self.fullSize:
                if self.randomClip:
                    rainy_depth = rainy_depth.crop((clipX,clipY,clipX+self.size,clipY+self.size))
                else:
                    rainy_depth = rainy_depth.resize((self.size, self.size), resample=self.interpolation)
            example["raw_rainy_depth"] = np.array(rainy_depth)
            example["rainy_depth"] = np.array(rainy_depth).astype(np.float32) / 127.5 - 1.0

        return example

class RainBlenderList(RainBlender):
    def __init__(self, json_file, data_root, size=None, interpolation="bicubic", fullSize=False, randomClip=True, useDepth=False):
        super().__init__(json_file, data_root, size, interpolation, fullSize, randomClip, useDepth)
            
    def getitem_fullsize(self, i):
        example = self.labels[i].copy()
        background = loadImgRGB(example["background_path"], False)
        rainy_image = loadImgRGB(example["rainy_image_path"], False)
        rain_layer = loadImgRGBA(example["rain_layer_path"], False)

        if self.useDepth:
            depth = loadImgGray(example["depth_path"], not self.randomClip)
            rainy_depth = loadImgGray(example["rainy_depth_path"], not self.randomClip)
                
        for size_idx, size_hw in enumerate(self.size):    
            factor = self.size[0] // size_hw
            size_new = (background.size[0] // factor, background.size[1] // factor)
            # Background
            background_reszie = background.resize(size_new, resample=self.interpolation)
            example["raw_background_%d" % size_idx] = np.array(background_reszie)
            example["background_%d" % size_idx] = np.array(background_reszie).astype(np.float32) / 127.5 - 1.0

            # Rainy Image
            rainy_image_reszie = rainy_image.resize(size_new, resample=self.interpolation)
            example["raw_rainy_image_%d" % size_idx] = np.array(rainy_image_reszie)
            example["rainy_image_%d" % size_idx] = np.array(rainy_image_reszie).astype(np.float32) / 127.5 - 1.0

            # Rain Layer
            rain_layer_reszie = rain_layer.resize(size_new, resample=self.interpolation)
            example["raw_rain_layer_%d" % size_idx] = np.array(rain_layer_reszie)
            rain_layer_reszie = np.array(rain_layer_reszie).astype(np.float32)
            example["rain_layer_%d" % size_idx] = rain_layer_reszie[...,:-1] / 127.5 - 1.0

            # Rain Mask
            mask = rain_layer_reszie[...,-1] /255.0
            mask[mask < 0.01] = 0
            mask[mask >= 0.01] = 1

            example["mask_%d" % size_idx] = mask
            mask = mask[...,None]

            masked_background = (1-mask)*background_reszie
            example["masked_background_%d"% size_idx] = masked_background / 127.5 - 1.0

            if self.useDepth:
                # Depth
                depth_reszie = depth.resize(size_new, resample=self.interpolation)
                example["raw_depth_%d" % size_idx] = np.array(depth_reszie)
                example["depth_%d" % size_idx] = np.array(depth_reszie).astype(np.float32) / 127.5 - 1.0
                
                # Rainy Depth
                rainy_depth_reszie = rainy_depth.resize(size_new, resample=self.interpolation)
                example["raw_rainy_depth_%d" % size_idx] = np.array(rainy_depth_reszie)
                example["rainy_depth_%d" % size_idx] = np.array(rainy_depth_reszie).astype(np.float32) / 127.5 - 1.0

        return example
            
    def __getitem__(self, i):
        if self.fullSize:
            return self.getitem_fullsize(i)
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
            mask[mask < 0.01] = 0
            mask[mask >= 0.01] = 1
            
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