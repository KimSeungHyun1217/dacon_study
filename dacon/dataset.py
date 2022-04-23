import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop, ColorJitter
import albumentations as A
import albumentations.pytorch


class AlbumAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.Resize(512,512),
            A.HorizontalFlip(p=0.5),
            #A.ColorJitter(brightness=0.2, contrast=0.2,p=0.5),
            #A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            A.pytorch.transforms.ToTensorV2()
        ])
        

    def __call__(self,image):
        return self.transform(image=image)


class DaconDataset(Dataset):
    
    def __init__(self,path):
        self.idledata = pd.read_csv(path)
        self.transform = None
        self.data_path = '/opt/ml/input/data/open/train/'
        self.num_class = len(self.idledata['label'].value_counts().index)
        self.wordtolabel = {}
        num = 0
        for i in self.idledata['label'].value_counts().index:
            self.wordtolabel[i] = num
            num += 1
        
        
    def get_num_class(self):
        return self.num_class
    

    def __len__(self):
        return len(self.idledata)

    def set_transform(self,transform):
        self.transform = transform

    def __getitem__(self,idx):
        image = Image.open(self.data_path + self.idledata.iloc[idx,1]).convert("RGB")
        image = np.array(image)

        label = self.wordtolabel[self.idledata.iloc[idx,4]]

        augment = self.transform()(image= image)

        image = augment['image']

        return image,label

class TestAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.Resize(512,512),
            A.Normalize(),
            A.pytorch.transforms.ToTensorV2()
        ])
        

    def __call__(self,image):
        return self.transform(image=image)

class TestDataset(Dataset):
    
    def __init__(self,path):
        self.idledata = pd.read_csv('/opt/ml/input/data/open/train_df.csv')
        self.transform = None
        self.data_path = '/opt/ml/input/data/open/test/'
        
        self.wordtolabel = {}
        num = 0
        for i in self.idledata['label'].value_counts().index:
            self.wordtolabel[num] = i 
            num += 1
        print(self.wordtolabel)
        self.testdata = pd.read_csv(path)

        
        
    def get_num_class(self):
        return self.num_class
    

    def __len__(self):
        return len(self.testdata)

    def set_transform(self,transform):
        self.transform = transform

    def __getitem__(self,idx):
        image = Image.open(self.data_path + self.testdata.iloc[idx,1]).convert("RGB")
        image = np.array(image)

        augment = self.transform()(image = image)

        image = augment['image']

        return image
