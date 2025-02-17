import os
import argparse
import yaml
import cv2
from natsort import natsorted
from tqdm import tqdm

# Read config from args and yaml file
def get_args():
    parser = argparse.ArgumentParser(description='CARLARain Video')

    parser.add_argument('--data_root',
                        type=str,
                        default=None,
                        help='The data root')    
    parser.add_argument('--types', nargs='+', help='Data types', required=True)
    parser.add_argument('--fps', type=str,default=10,help='FPS')
    parser.add_argument('--resolution', type=str,default="[2048,1024]",help='Resolution')
    
    args = parser.parse_args()

    return args

def merge_image_to_video(inPath,outPath,fps,img_size):
    files = natsorted(os.listdir(inPath))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(outPath, fourcc, fps, img_size)
    for filename in tqdm(files):
        filePath = os.path.join(inPath, filename)
        frame = cv2.imread(filePath)
        video.write(frame)
    video.release()



if __name__=="__main__":
    args = get_args()


    for type in args.types:
        print("Merging Video {} in {}...".format(type,args.data_root))
        merge_image_to_video(os.path.join(args.data_root,type),
                             os.path.join(args.data_root,type+".mp4"),
                             args.fps,eval(args.resolution))
