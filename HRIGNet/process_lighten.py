import os
from PIL import Image
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    dataRoot = "/home/zhoukaibin/code/CARLRain/data/seqTownsCombine/test"
    bgRoot = os.path.join(dataRoot,"background")
    rainyRoot = os.path.join(dataRoot,"rainy_wind_0.3_50mm")
    outRoot = os.path.join(dataRoot,"rainy_wind_0.3_50mm_lighten")
    os.makedirs(outRoot,exist_ok=True)

    for file in tqdm(os.listdir(bgRoot)):
        bgImg = Image.open(os.path.join(bgRoot,file))
        rainyImg = Image.open(os.path.join(rainyRoot,file))

        outImg = Image.fromarray(np.max((np.array(bgImg),np.array(rainyImg)),axis=0))
        outImg.save(os.path.join(outRoot,file),quality=60)

