import os

join = os.path.join

import torch
from torch.utils.data import Dataset
import numpy as np

from .datainfo import *


class TaskDataset(Dataset):
    def __init__(self, data_root, train=True):
        self.data_root = data_root
        self.train = train

        files = sorted(os.listdir(join(data_root, "npy_gts")))
        self.file_names = [join(data_root, "npy_gts", f) for f in files]


    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img = np.load(self.file_names[index].replace("npy_gts", "npy_imgs")).transpose(2, 0, 1)
        gt = np.load(self.file_names[index])

        y_indices, x_indices = np.where(gt > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        if self.train:
            H, W = gt.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
           
        box = np.array([x_min, y_min, x_max, y_max])
       
        data = {
            "img": torch.tensor(img).float(),
            "box": torch.tensor(box).float(),
            "name": self.file_names[index],
        }
     
        return data,torch.tensor(gt[None, :, :]).long()


class GeneralDataset(Dataset):
    def __init__(self, data_root, train=True):
        self.data_root = data_root
        self.train = train

        self.file_names = []
        # self.task_names = []
        if "npy_gts" in os.listdir(data_root):
                    files = sorted(os.listdir(join(data_root, "npy_gts")))
                    self.file_names += [join(data_root, "npy_gts", f) for f in files]
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img = np.load(self.file_names[index].replace("npy_gts", "npy_imgs")).transpose(2, 0, 1)
        img1 = np.rot90(img, k=-1, axes=(1, 2))
        img2 = np.rot90(img, k=-2, axes=(1, 2))
        img3 = np.rot90(img, k=-3, axes=(1, 2))
        gt = np.load(self.file_names[index])

        y_indices, x_indices = np.where(gt > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        if self.train:
            H, W = gt.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
            #顺时针90度box
            x_min1 = H-y_max
            x_max1 = H-y_min
            y_min1 = x_min
            y_max1 = x_max
            #顺时针180度box
            x_min2 = W-x_max
            x_max2 = W-x_min
            y_min2 = H-y_max
            y_max2 = H-y_min
            #顺时针270度box
            x_min3 = y_min
            x_max3 = y_max
            y_min3 = W-x_max
            y_max3 = W-x_min
        box = np.array([x_min, y_min, x_max, y_max])
        box1 = np.array([x_min1, y_min1, x_max1, y_max1])
        box2 = np.array([x_min2, y_min2, x_max2, y_max2])
        box3 = np.array([x_min3, y_min3, x_max3, y_max3])
        data = {
            "img": torch.tensor(img).float(),
            "box": torch.tensor(box).float(),
            "name": self.file_names[index],
        }
        data1 = {
            "img": torch.tensor(img1.copy()).float(),
            "box": torch.tensor(box1).float(),
            "name": self.file_names[index],
        }
        data2 = {
            "img": torch.tensor(img2.copy()).float(),
            "box": torch.tensor(box2).float(),
            "name": self.file_names[index],
        }
        data3 = {
            "img": torch.tensor(img3.copy()).float(),
            "box": torch.tensor(box3).float(),
            "name": self.file_names[index],
        }
        return data, data1,data2,data3,torch.tensor(gt[None, :, :]).long()

class GeneraltestDataset(Dataset):
    def __init__(self, data_root, train=True):
        self.data_root = data_root
        self.train = train

        self.file_names = []
        # self.task_names = []
        if "npy_gts" in os.listdir(data_root):
                    files = sorted(os.listdir(join(data_root, "npy_gts")))
                    self.file_names += [join(data_root, "npy_gts", f) for f in files]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img = np.load(self.file_names[index].replace("npy_gts", "npy_imgs")).transpose(2, 0, 1)
        gt = np.load(self.file_names[index])

        y_indices, x_indices = np.where(gt > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        if self.train:
            H, W = gt.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        box = np.array([x_min, y_min, x_max, y_max])

        data = {
            "img": torch.tensor(img).float(),
            "box": torch.tensor(box).float(),
            "name": self.file_names[index],
        }
        return data, torch.tensor(gt[None, :, :]).long()

class TaskFinetuneDataset(Dataset):
    def __init__(self, data_root, train=True):
        self.data_root = data_root
        self.train = train

        files = sorted(os.listdir(join(data_root, "npy_gts")))
        self.file_names = [join(data_root, "npy_gts", f) for f in files]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img = np.load(self.file_names[index].replace("npy_gts", "npy_imgs")).transpose(2, 0, 1)
        gt = np.load(self.file_names[index])

        y_indices, x_indices = np.where(gt > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        if self.train:
            H, W = gt.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        box = np.array([x_min, y_min, x_max, y_max])

        data = {
            "img": torch.tensor(img).float(),
            "box": torch.tensor(box).float(),
            "name": self.file_names[index],
        }
        return data, torch.tensor(gt[None, :, :]).long()


class DatasetDataset(Dataset):
    def __init__(self, data_root, train=True):
        self.data_root = data_root
        self.train = train

        self.file_names = []
        self.task_names = []
        for task in sorted(os.listdir(data_root)):
            task_dir = join(data_root, task)
            if "npy_gts" in os.listdir(task_dir):
                files = sorted(os.listdir(join(task_dir, "npy_gts")))
                self.file_names += [join(task_dir, "npy_gts", f) for f in files]
                self.task_names += [task] * len(files)
            else:
                for sequence in os.listdir(task_dir):
                    sequence_dir = join(task_dir, sequence)
                    files = sorted(os.listdir(join(sequence_dir, "npy_gts")))
                    self.file_names += [join(sequence_dir, "npy_gts", f) for f in files]
                    self.task_names += [task] * len(files)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img = np.load(self.file_names[index].replace("npy_gts", "npy_imgs")).transpose(2, 0, 1)
        gt = np.load(self.file_names[index])

        y_indices, x_indices = np.where(gt > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        if self.train:
            H, W = gt.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        box = np.array([x_min, y_min, x_max, y_max])

        data = {
            "img": torch.tensor(img).float(),
            "box": torch.tensor(box).float(),
            "name": self.file_names[index],
        }
        return data, torch.tensor(gt[None, :, :]).long()
