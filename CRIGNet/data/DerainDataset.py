import os
import os.path
import json
import numpy as np
import torch
import cv2
import torch.utils.data as udata
from numpy.random import RandomState

class TrainDataset(udata.Dataset):
    def __init__(self, inputname, gtname,patchsize,length):
        super().__init__()
        self.patch_size=patchsize
        self.input_dir = os.path.join(inputname)
        self.gt_dir = os.path.join(gtname)
        self.img_files = os.listdir(self.input_dir)
        self.rand_state = RandomState(66)
        self.file_num = len(self.img_files)
        self.sample_num = length
    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        file_name = self.img_files[idx % self.file_num]
        img_file = os.path.join(self.input_dir, file_name)
        O = cv2.imread(img_file)
        b, g, r = cv2.split(O)
        input_img = cv2.merge([r, g, b])
        O,row,col= self.crop(input_img)
        O = O.astype(np.float32) / 255
        O = np.transpose(O, (2, 0, 1))

        gt_file = os.path.join(self.gt_dir, "no"+file_name)
        B = cv2.imread(gt_file)
        b, g, r = cv2.split(B)
        gt_img = cv2.merge([r, g, b])
        B = gt_img[row: row + self.patch_size, col : col + self.patch_size]
        B = B.astype(np.float32) / 255
        B = np.transpose(B, (2, 0, 1))
        return torch.Tensor(O), torch.Tensor(B)

    def crop(self, img):
        h, w, c = img.shape
        p_h, p_w = self.patch_size, self.patch_size
        if p_h >= h or p_w >= w:
            r, c = 0,0
        else:
            r = self.rand_state.randint(0, h - p_h)
            c = self.rand_state.randint(0, w - p_w)
        O = img[r: r + p_h, c : c + p_w]
        return O,r,c

class ValDataset(udata.Dataset):
    def __init__(self, inputname,gtname):
        super().__init__()
        self.input_dir = os.path.join(inputname)
        self.gt_dir = os.path.join(gtname)
        self.img_files = os.listdir(self.input_dir)
        self.file_num = len(self.img_files)
    def __len__(self):
        return int(self.file_num)

    def __getitem__(self, idx):
        file_name = self.img_files[idx % self.file_num]
        img_file = os.path.join(self.input_dir, file_name)
        O = cv2.imread(img_file)
        b, g, r = cv2.split(O)
        input_img = cv2.merge([r, g, b])
        O = input_img.astype(np.float32) / 255
        O = np.transpose(O, (2, 0, 1))

        gt_file = os.path.join(self.gt_dir, file_name)
        B = cv2.imread(gt_file)
        b, g, r = cv2.split(B)
        gt_img = cv2.merge([r, g, b])
        B = gt_img.astype(np.float32) / 255
        B = np.transpose(B, (2, 0, 1))
        return torch.Tensor(O), torch.Tensor(B)

class SPATrainDataset(udata.Dataset):
    def __init__(self, dir, sub_files, patchSize,sample_num,train_num):
        super().__init__()
        self.dir = dir
        self.patch_size = patchSize
        self.sample_num= sample_num
        self.train_num = train_num
        self.sub_files=sub_files
        self.rand_state = RandomState(66)
    def __len__(self):
        return self.sample_num
    def __getitem__(self, idx):
        file_name = self.sub_files[idx % int(self.train_num)]
        input_file_name = file_name.split(' ')[0]
        gt_file_name = file_name.split(' ')[1][:-1]

        O = cv2.imread(self.dir+input_file_name)
        b, g, r = cv2.split(O)
        input_img = cv2.merge([r, g, b])
        O,row,col= self.crop(input_img)
        O = O.astype(np.float32) / 255
        O = np.transpose(O, (2, 0, 1))

        B = cv2.imread(self.dir+ gt_file_name)
        b, g, r = cv2.split(B)
        gt_img = cv2.merge([r, g, b])
        B = gt_img[row: row + self.patch_size, col : col + self.patch_size]
        B = B.astype(np.float32) / 255
        B = np.transpose(B, (2, 0, 1))
        return torch.Tensor(O), torch.Tensor(B)

    def crop(self, img):
        h, w, c = img.shape
        p_h, p_w = self.patch_size, self.patch_size
        if p_h >= h or p_w >= w:
            r, c = 0,0
        else:
            r = self.rand_state.randint(0, h - p_h)
            c = self.rand_state.randint(0, w - p_w)
        O = img[r: r + p_h, c : c + p_w]
        return O,r,c


class CRIGTrainDataset(udata.Dataset):
    def __init__(self, path_real,patchsize,length,
                with_render=False,
                path_render=None,
                max_intensity = 100,
                zero_std=False,
                use_padding=False):
        super().__init__()
        self.patch_size=patchsize
        self.path_real = path_real
        with open(os.path.join(path_real,"dataset.json"),"r") as f:
            self.dataset_real = json.load(f)
        self.rand_state = RandomState(66)
        self.sample_num = length
        # self.sample_num = len(self.dataset_real)
        self.file_num = len(self.dataset_real)
        self.with_render = with_render
        if with_render:
            self.path_render = path_render
            self.max_intensity = max_intensity
            with open(os.path.join(path_render,"trainset.json"),"r") as f:
                self.dataset_render = json.load(f)
        self.zero_std = zero_std
        self.use_padding = use_padding

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        if self.use_padding:
            return self.getitem_padding(idx)
        else:
            return self.getitem_crop(idx)

    
    def getitem_crop(self,idx):
        file_idx = idx % self.file_num
        img_file = os.path.join(self.path_real, self.dataset_real[file_idx]["input"])
        O = cv2.imread(img_file)
        b, g, r = cv2.split(O)
        input_img = cv2.merge([r, g, b])
        O,row,col= self.crop(input_img)
        O = O.astype(np.float32) / 255
        O = np.transpose(O, (2, 0, 1))

        gt_file = os.path.join(self.path_real, self.dataset_real[file_idx]["label"])
        B = cv2.imread(gt_file)
        b, g, r = cv2.split(B)
        gt_img = cv2.merge([r, g, b])
        B = gt_img[row: row + self.patch_size, col : col + self.patch_size]
        B = B.astype(np.float32) / 255
        B = np.transpose(B, (2, 0, 1))

        if self.with_render:
            render_idx = self.rand_state.randint(0, len(self.dataset_render))
            img_render_file = os.path.join(self.path_render, self.dataset_render[render_idx]["rainy_image"])
            R_render = cv2.imread(img_render_file)
            b, g, r = cv2.split(R_render)
            render_img = cv2.merge([r, g, b])
            R_render,row,col= self.crop(render_img)
            R_render = R_render.astype(np.float32) / 255
            R_render = np.transpose(R_render, (2, 0, 1))

            img_mask_file = os.path.join(self.path_render, self.dataset_render[render_idx]["rain_mask"])
            R_mask = cv2.imread(img_mask_file,cv2.IMREAD_GRAYSCALE)
            R_mask = R_mask[row: row + self.patch_size, col : col + self.patch_size]
            R_mask[R_mask>=127] = 255
            R_mask[R_mask<127] = 0
            R_mask = R_mask.astype(np.float32) / 255

            label = np.array([ 
                self.dataset_render[render_idx]["intensity"]/self.max_intensity,
                (self.dataset_render[render_idx]["wind"]+1.0)/2.0,
            ]).astype(np.float32)
            if self.zero_std:
                label = label * 2.0 - 1.0

            return torch.Tensor(O), torch.Tensor(B), torch.Tensor(R_render), torch.Tensor(label), torch.Tensor(R_mask)
        else:
            return torch.Tensor(O), torch.Tensor(B)

    def getitem_padding(self,idx):
        file_idx = idx % self.file_num
        img_file = os.path.join(self.path_real, self.dataset_real[file_idx]["input"])
        O = cv2.imread(img_file)
        O = self.padding(O)
        b, g, r = cv2.split(O)
        O = cv2.merge([r, g, b])
        O = O.astype(np.float32) / 255
        O = np.transpose(O, (2, 0, 1))

        gt_file = os.path.join(self.path_real, self.dataset_real[file_idx]["label"])
        B = cv2.imread(gt_file)
        b, g, r = cv2.split(B)
        gt_img = cv2.merge([r, g, b])
        B = self.padding(gt_img)
        B = B.astype(np.float32) / 255
        B = np.transpose(B, (2, 0, 1))

        if self.with_render:
            render_idx = self.rand_state.randint(0, len(self.dataset_render))
            img_render_file = os.path.join(self.path_render, self.dataset_render[render_idx]["rainy_image"])
            R_render = cv2.imread(img_render_file)
            R_render = self.padding(R_render)
            b, g, r = cv2.split(R_render)
            R_render = cv2.merge([r, g, b])
            R_render = R_render.astype(np.float32) / 255
            R_render = np.transpose(R_render, (2, 0, 1))

            img_mask_file = os.path.join(self.path_render, self.dataset_render[render_idx]["rain_mask"])
            R_mask = cv2.imread(img_mask_file,cv2.IMREAD_GRAYSCALE)
            R_mask = self.padding(R_mask)
            R_mask[R_mask>=127] = 255
            R_mask[R_mask<127] = 0
            R_mask = R_mask.astype(np.float32) / 255

            label = np.array([ 
                self.dataset_render[render_idx]["intensity"]/self.max_intensity,
                (self.dataset_render[render_idx]["wind"]+1.0)/2.0,
            ]).astype(np.float32)
            if self.zero_std:
                label = label * 2.0 - 1.0

            return torch.Tensor(O), torch.Tensor(B), torch.Tensor(R_render), torch.Tensor(label), torch.Tensor(R_mask)
        else:
            return torch.Tensor(O), torch.Tensor(B)

    def crop(self, img):
        h, w, c = img.shape
        p_h, p_w = self.patch_size, self.patch_size
        if p_h >= h or p_w >= w:
            r, c = 0,0
        else:
            r = self.rand_state.randint(0, h - p_h)
            c = self.rand_state.randint(0, w - p_w)
        O = img[r: r + p_h, c : c + p_w]
        return O,r,c

    def padding(self, img):
        if len(img.shape) == 3:
            h, w, _ = img.shape
        elif len(img.shape) == 2:
            h, w = img.shape
        p_h, p_w = self.patch_size-h, self.patch_size-w
        O = cv2.copyMakeBorder(img, 0, p_h, 0, p_w, cv2.BORDER_REFLECT)
        return O
    
class RenderDataset(udata.Dataset):
    def __init__(self,patchsize,
                path_render,
                max_intensity = 100,
                zero_std=False):
        super().__init__()
        self.patch_size=patchsize
        self.rand_state = RandomState(824)
        self.path_render = path_render
        self.max_intensity = max_intensity
        with open(os.path.join(path_render,"trainset.json"),"r") as f:
            self.dataset_render = json.load(f)
        self.sample_num = len(self.dataset_render)
        self.zero_std = zero_std
        self.get_labels()
    
    def get_labels(self):
        intensitys = set()
        winds = set()
        self.labels = []
        self.label_dict = {}
        for idx in range(len(self.dataset_render)):
            intensity = round(self.dataset_render[idx]["intensity"]/self.max_intensity,2)
            wind = round((self.dataset_render[idx]["wind"]+1.0)/2.0,2)
            if self.zero_std:
                intensity = intensity * 2.0 - 1.0
                wind = wind * 2.0 - 1.0
            self.labels.append([ intensity,wind ])
            intensitys.add(intensity)
            winds.add(wind)
            key = '{:.2f}_{:.2f}'.format(intensity,wind)
            if key in self.label_dict:
                self.label_dict[key].append(idx)
            else:
                self.label_dict[key] = [idx]
        self.label_set = [ intensitys,winds ]
        self.labels = np.array(self.labels).astype(np.float32)
    
    def get_random_label(self,label_idx,radom_size=1):
        labels = self.rand_state.choice(list(self.label_set[label_idx]),radom_size)
        return labels

    def get_samples_by_label(self,labels,sample_size=1):
        samples = []
        for label in labels:
            key = '{:.2f}_{:.2f}'.format(label[0],label[1])
            sample = self.rand_state.choice(self.label_dict[key],size=sample_size)
            samples.extend(sample)
        return samples

    def getitem_by_indices(self,idxs):
        R_renders = []
        labels = []
        R_masks = []
        for idx in idxs:
            R_render, label, R_mask = self.__getitem__(idx)
            R_renders.append(R_render)
            labels.append(label)
            R_masks.append(R_mask)

        R_renders = torch.stack(R_renders,dim=0)
        labels = torch.stack(labels,dim=0)
        R_masks = torch.stack(R_masks,dim=0)
        return R_renders, labels, R_masks

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        img_render_file = os.path.join(self.path_render, self.dataset_render[idx]["rainy_image"])
        R_render = cv2.imread(img_render_file)
        b, g, r = cv2.split(R_render)
        render_img = cv2.merge([r, g, b])
        R_render,row,col= self.crop(render_img)
        R_render = R_render.astype(np.float32) / 255
        R_render = np.transpose(R_render, (2, 0, 1))

        img_mask_file = os.path.join(self.path_render, self.dataset_render[idx]["rain_mask"])
        R_mask = cv2.imread(img_mask_file,cv2.IMREAD_GRAYSCALE)
        R_mask = R_mask[row: row + self.patch_size, col : col + self.patch_size]
        R_mask[R_mask>=127] = 255
        R_mask[R_mask<127] = 0
        R_mask = R_mask.astype(np.float32) / 255

        label = self.labels[idx]

        return torch.Tensor(R_render), torch.Tensor(label), torch.Tensor(R_mask)

    def crop(self, img):
        h, w, c = img.shape
        p_h, p_w = self.patch_size, self.patch_size
        if p_h >= h or p_w >= w:
            r, c = 0,0
        else:
            r = self.rand_state.randint(0, h - p_h)
            c = self.rand_state.randint(0, w - p_w)
        O = img[r: r + p_h, c : c + p_w]
        return O,r,c
    

class CRIGSPATrainDataset(udata.Dataset):
    def __init__(self, data_path, patchSize,sample_num,train_num):
        super().__init__()
        self.data_path = data_path
        self.patch_size = patchSize
        self.sample_num= sample_num
        self.train_num = train_num
        with open(os.path.join(data_path,"dataset.json"),"r") as f:
            sub_files = json.load(f)
            self.sub_files = sub_files[:int(train_num)]
        self.rand_state = RandomState(66)
    def __len__(self):
        return self.sample_num
    def __getitem__(self, idx):
        sample = self.sub_files[idx % int(self.train_num)]
        input_file_name = sample["input"]
        gt_file_name = sample["label"]

        O = cv2.imread(os.path.join(self.data_path,input_file_name))
        b, g, r = cv2.split(O)
        input_img = cv2.merge([r, g, b])
        O,row,col= self.crop(input_img)
        O = O.astype(np.float32) / 255
        O = np.transpose(O, (2, 0, 1))

        B = cv2.imread(os.path.join(self.data_path,gt_file_name))
        b, g, r = cv2.split(B)
        gt_img = cv2.merge([r, g, b])
        B = gt_img[row: row + self.patch_size, col : col + self.patch_size]
        B = B.astype(np.float32) / 255
        B = np.transpose(B, (2, 0, 1))
        return torch.Tensor(O), torch.Tensor(B)

    def crop(self, img):
        h, w, c = img.shape
        p_h, p_w = self.patch_size, self.patch_size
        if p_h >= h or p_w >= w:
            r, c = 0,0
        else:
            r = self.rand_state.randint(0, h - p_h)
            c = self.rand_state.randint(0, w - p_w)
        O = img[r: r + p_h, c : c + p_w]
        return O,r,c