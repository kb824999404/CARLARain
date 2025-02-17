# Instance Segmentation To Object Detection Dataset
import os
import argparse
import yaml
from config import Config

import cv2
import numpy as np
import json
from tqdm import tqdm

from labels_cityscapes import id2label

import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from multiprocessing import Pool, cpu_count

# Read config from args and yaml file
def get_args():
    parser = argparse.ArgumentParser(description='CARLA Run')

    parser.add_argument('-c', '--config',
                        type=str,
                        default=None,
                        help='The config file')    
    
    args = parser.parse_args()

    if args.config:
        assert os.path.exists(args.config), ("The config file is missing.", args.config)
        with open(args.config,"r") as f:
            cfg = yaml.safe_load(f)['scene']
        for key in cfg:
            args.__dict__[key] = cfg[key]

    return args

def get_objects(isMap):
    H, W = isMap.shape[:2]
    objects = {}
    for y in range(H):
        for x in range(W):
            tag, id1, id2 = isMap[y][x]
            id = f"{id1}-{id2}"
            if id not in objects:
                objects[id] = {
                    "tag": tag,
                    "pixels": []
                }
            objects[id]["pixels"].append((y, x))
        
    return objects
    
def get_data(params):
    index,isFile = params
    fileName = isFile.split(".")[0]
    if os.path.exists(os.path.join(odRoot, fileName + ".json")):
        return
    
    start_time = time.time()
    print(f"Process item #{index}/{len(isMaps)}")
    isMap = cv2.imread(os.path.join(isRoot, isFile), cv2.IMREAD_COLOR)[:, :, ::-1]
    H, W = isMap.shape[:2]
    
    # 获取物体
    # print(f"Geting objects for {isFile} ...")
    objects = get_objects(isMap)

    # 构造每个物体的polygon
    # print(f"Creating polygon for {isFile} ...")
    objects_polygon = []
    for obj_id, obj_info in objects.items():
        tag = obj_info["tag"]
        pixels = obj_info["pixels"]
        
        if tag not in id2label:
            continue
        label = id2label[tag].name

        # 使用凸包算法来构造近似的多边形
        hull = cv2.convexHull(np.array(pixels, dtype=np.int32))

        polygon = []
        for point in hull:
            polygon.append(point[0].tolist())

        obj = {
            "label": label,
            "polygon": polygon
        }
        objects_polygon.append(obj)

    detectionInfo = {
        "imgHeight": H,
        "imgWidth": W,
        "objects": objects_polygon
    }

    with open(os.path.join(odRoot, fileName + ".json"), "w") as f:
        json.dump(detectionInfo, f)
        # print("Save to "+os.path.join(odRoot, fileName + ".json"))
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"#{index}/{len(isMaps)} Execution time: {execution_time} seconds")
        
if __name__=="__main__":
    args = get_args()
    seqRoot = os.path.join(args.dataRoot,args.name)
    isRoot = os.path.join(seqRoot,Config.cameraPaths["is"])
    odRoot = os.path.join(seqRoot,"object_detection")
    os.makedirs(odRoot,exist_ok=True)

    isMaps = [  item for item in os.listdir(isRoot) if item.split('.')[-1] == "png" ]
    params = [ (i,item) for i,item in enumerate(isMaps)  ]
    
    # for isFile in tqdm(isMaps):
    #     get_data(isFile)
        
    num_processes = cpu_count()  # 获取CPU核心数，确定进程数量
    with Pool(num_processes) as pool:
        pool.map(get_data, params)
    # with ThreadPoolExecutor(max_workers=20) as executor:
    #     print("Creating threads..")
    #     futures = [executor.submit(get_data, isFile) for isFile in isMaps]
    #     print("Create threads done.")
    #     for future in tqdm(as_completed(futures), total=len(futures)):
    #         try:
    #             future.result()
    #         except Exception as e:
    #             print(f"Error: {e}")


# Cityscapes标注格式
# {
#   "imgHeight": 1024, 
#   "imgWidth": 2048, 
#   "objects": [
#       {
#         "label": "road", 
#         "polygon": [
#             [
#                 0, 
#                 769
#             ], 
#             [
#                 290, 
#                 574
#             ],
#             // ... n polygon coordinates
#         ]
#       },
#       // ... n objects in this image
#   ]
# }