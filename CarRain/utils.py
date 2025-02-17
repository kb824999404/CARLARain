import os, platform
import time
import shutil

def create_dir_not_exist(path):
    sys = platform.system()
    if sys == "Windows":
        for length in range(0, len(path.split(os.path.sep))):
            check_path = os.path.sep.join(path.split(os.path.sep)[:(length+1)])
            if not os.path.exists(check_path):
                os.mkdir(check_path)
                print(f'Created Dir: {check_path}')
    elif sys == "Linux":
        if not os.path.exists(path):
            os.system(f'mkdir -p {path}')
            print(f'Created Dir: {path}')

def rename_dir_existed(path):
    filemt = time.localtime(os.stat(path).st_mtime)
    path_new = path + time.strftime("_%Y_%m_%d_%H_%M_%S", filemt)
    shutil.copytree(path,path_new)
    shutil.rmtree(path)


def printDivider(char="=",length=50):
    print(char*length)
