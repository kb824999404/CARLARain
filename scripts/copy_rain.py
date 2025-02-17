import os
import shutil

if __name__=="__main__":
    rainRoot="/home/zhoukaibin/data7/dataset/rain-mask-wind-single/rain_512"
    outRoot="/home/zhoukaibin/data7/code/CARLRain/data/seqTest/rain"
    for wind in os.listdir(rainRoot):
        windRoot = os.path.join(rainRoot,wind)
        if os.path.isdir(windRoot):
            for intensity in os.listdir(windRoot):
                intensityRoot = os.path.join(windRoot,intensity,"rainy_image")
                for img in os.listdir(intensityRoot):
                    target = "{}_{}_{}".format(wind,intensity,img)
                    shutil.copyfile(os.path.join(intensityRoot,img),os.path.join(outRoot,target))
                    print(target)
