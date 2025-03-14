import os
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor,as_completed

def combine_dataset(data_root,out_root,scenes):
    data_types = [ "background","rainy","semantic_segmentation","semantic_segmentation_train","semantic_segmentation_eval","instance_segmentation","depth"  ]
    for data_type in data_types:
        os.makedirs(os.path.join(out_root,data_type),exist_ok=True)
    for scene in tqdm(scenes):
        for data_type in data_types:
            print(scene,data_type)
            for file in tqdm(os.listdir(os.path.join(data_root,scene,data_type))):
                source_path = os.path.join(data_root,scene,data_type,file)
                target_path = os.path.join(out_root,data_type,f"{scene}_{file}")
                shutil.copyfile(source_path,target_path)

def combine_dataset_multi(params):
    data_root,out_root,scene,data_type = params
    print("Start:",scene,data_type)
    for file in os.listdir(os.path.join(data_root,scene,data_type)):
        source_path = os.path.join(data_root,scene,data_type,file)
        target_path = os.path.join(out_root,data_type,f"{scene}_{file}")
        if os.path.isfile(source_path) and not os.path.exists(target_path):
            shutil.copyfile(source_path,target_path)
    print("Done:",scene,data_type)

def main():
    data_root = "/home/zhoukaibin/code/CARLRain/data"
    out_root = os.path.join(data_root,"seqTownsCombine")
    os.makedirs(out_root,exist_ok=True)

    maps_train = [ f"seqTown{i:02d}" for i in [ 1,2,3,4,5,6,7  ] ]
    maps_test = [ f"seqTown{i:02d}" for i in [ 10  ] ]
    suffixs = [  "Clear","ClearSunset","ClearNight"  ]
    scenes_train = [ f"{m}{s}" for m in maps_train for s in suffixs  ]
    scenes_test = [ f"{m}{s}" for m in maps_test for s in suffixs  ]
    train_root = os.path.join(out_root,"train")
    test_root = os.path.join(out_root,"test")
    os.makedirs(train_root,exist_ok=True)
    os.makedirs(test_root,exist_ok=True)

    # data_types = [ "background","rainy","semantic_segmentation","semantic_segmentation_train","semantic_segmentation_eval","instance_segmentation","depth"  ]
    data_types = [ "background"  ]

    params = [ ]
    for scene in scenes_train:
        for data_type in data_types:
            params.append((data_root,train_root,scene,data_type))
            os.makedirs(os.path.join(train_root,data_type),exist_ok=True)
    # for scene in scenes_test:
    #     for data_type in data_types:
    #         params.append((data_root,test_root,scene,data_type))
    #         os.makedirs(os.path.join(test_root,data_type),exist_ok=True)
    
    # 多进程
    # num_processes = 20
    # with Pool(num_processes) as pool:
        # pool.map(combine_dataset_multi, params)
    # 多线程
    num_workers = 20
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        print("Creating threads..")
        futures = [executor.submit(combine_dataset_multi, p) for p in params]
        print("Create threads done.")
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"Error: {e}")

    # combine_dataset(data_root,train_root,scenes_train)
    # combine_dataset(data_root,test_root,scenes_test)

if __name__=="__main__":
    main()
