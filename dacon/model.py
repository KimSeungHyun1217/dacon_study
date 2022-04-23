import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import reduce
import torch


class PretrainedModel:
    def __init__(self, model_name,num_class,pretrain=True):
        self.model_name = model_name
        self.num_class = num_class
        model = getattr(models,model_name)(pretrained=pretrain)
        
        for name, mod in reversed(list(model.named_modules())):
            if isinstance(mod, nn.Linear):
                mod_path = name.split('.')
                classifier_parent = reduce(nn.Module.children, mod_path[:-1], model)
                setattr(classifier_parent, mod_path[-1], nn.Sequential(
                    nn.Linear(mod.in_features, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.7),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, self.num_class)
                ))
                break

        self.model = model
        
    def __call__(self):
        return self.model