from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image
import numpy as np
import pandas as pd



class CustomTransform:
    def __init__(self):
        self.transforms = A.Compose([
            A.Resize(64,224),
            A.Blur(p=0.5),
            A.Normalize(),
            ToTensorV2()
        ])

    def __call__(self,image):
        return self.transforms(image = image)
    

class CustomDataset(Dataset):
    def __init__(self,df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def set_transform(self,transform):
        self.transform = transform

    def __getitem__(self,idx):
        image = Image.open(self.df['img_path'][idx]).convert("RGB")
        image = np.array(image)

        label = self.df['label'][idx]
        
        augment = self.transform()(image=image)
        image = augment['image']
        
        return image,label