import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset , TestAugmentation

from model import PretrainedModel




@torch.no_grad()
def inference():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = PretrainedModel('resnet50', 88 ,pretrain=False)()
    model.load_state_dict(torch.load('/opt/ml/code/dacon/model/latest.pth'))
    model = model.to(device)
    
    model.eval()

    info = pd.read_csv('/opt/ml/input/data/open/sample_submission.csv')
    

    dataset = TestDataset('/opt/ml/input/data/open/test_df.csv')
    dataset.set_transform(TestAugmentation)

    loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=2,
        shuffle=False,
    )
    num1 = 0
    
    preds = []
    with torch.no_grad():
        for idx, data in enumerate(loader):
            images = data
            images = images.to(device)
            pred = model(images)
            
            pred = torch.argmax(pred, dim=-1)
            
            for i in pred:
                preds.append(dataset.wordtolabel[int(i)])
                num1 += 1
            print(num1)
    
    print(len(preds))
    print(len(info))

    info['label'] = preds
    info.to_csv('/opt/ml/input/data/open/inference.csv', index=False)

    print(f'Inference Done!')


if __name__ == '__main__':
    inference()