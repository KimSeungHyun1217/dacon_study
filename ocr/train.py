import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from dataset import CustomDataset,CustomTransform
from converter import LabelConverter
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    
    
    df = pd.read_csv("./train.csv")

    df['len'] = df['label'].str.len()
    train_v1 = df[df['len']==1]

    df = df[df['len']>1]
    train_v2, val = train_test_split(df, test_size=0.2, random_state=1313)

    train = pd.concat([train_v1, train_v2])

    train_gt = [gt for gt in train['label']]
    train_gt = "".join(train_gt)
    letters = sorted(list(set(list(train_gt))))
    
    print(len(train),len(val))
    
    train_dataset = CustomDataset(train.reset_index())
    train_dataset.set_transform(CustomTransform)

    val_dataset = CustomDataset(val.reset_index())
    val_dataset.set_transform(CustomTransform)

    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = 64, shuffle=True)

    attn = LabelConverter(letters)
    
    model = Model(num_class = len(attn.character))
    model = model.to(device)
    model.train()
    
    
    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    criterion = criterion.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1, rho=0.95, eps=1e-8)
    best_cost = 9999999
    
    for i in range(20):
        model.train()
        for image_tensors , labels in train_loader:
            image = image_tensors.to(device)
            text, length = attn.encode(labels, batch_max_length=5)
            batch_size = image.size(0)

            preds = model(image, text[:, :-1]) 
            target = text[:, 1:]
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            print(1)
        model.eval()
        total_cost = 0 
        with torch.no_grad():
            for image_tensors,labels in val_loader:
                image = image_tensors.to(device)
                batch_size = image.size(0)
                
                text_for_loss,length_for_loss = converter.encode(labels,batch_max_length = 5)
                
                length_for_pred = torch.IntTensor([5] * batch_size).to(device)
                text_for_pred = torch.LongTensor(batch_size, 5 + 1).fill_(0).to(device)
                
                preds = model(image,text_for_pred,is_train=False)
                preds = preds[:, :text_for_loss.shape[1] - 1, :]
                target = text_for_loss[:, 1:]
                cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
                
                total_cost += cost
        
        print(cost)
        if best_cost > cost:
            torch.save(model.state_dict(), f'./saved_models/test/best_accuracy.pth')
            

            
if __name__ == '__main__':
    train()