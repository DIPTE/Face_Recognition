#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：L5 -> train
@IDE    ：PyCharm
@Author ：DIPTE
@Contect : 2563640356@qq.com
@Date   ：2020/8/2 2:21
@Desc   ：
=================================================='''
import os, sys

sys.path.append('.')
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from torchsummary import summary

from LFW_CASIAWebFace_Dataset import CASIAWebFace, LFW,SiameseCASIAWebFace,TripletFaceDataset
from SEResNet_IR import ResNet18, ResNet34
# from trainer import Trainer
# from Contrastive_trainer import Trainer##using Contrastive
from Triplet_trainer import Trainer##using Triplet

from loss_function import NormFace, SphereFace, CosFace, ArcFace
from modified_resnet import resnet18, resnet34
from loss_function import *

# margin type
margin_type = 'NormFace'
feature_dim = 512
#criterion set
criterion='Triplet'
print(margin_type,criterion)
#Softmax、NormFace、SphereFace、CosFace、ArcFace、Contrastive、Triplet、OHEM、FocalLoss

# training config
BATCH_SIZE = 8#16#32#64#128#256
LR = 1e-3
EPOCHS = 20
STEPS = [4, 10, 18]
USE_CUDA = True


# Set training device
use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
torch.backends.cudnn.benchmark = True
print(device)


# Set dataloader
transforms = tfs.Compose([
    tfs.ToPILImage(),
    tfs.Resize((112, 112)),###resize
    tfs.ToTensor(),
    tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])


kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
train_data = TripletFaceDataset(root_dir='CASIA-maxpy-clean', csv_name='CASIAWebFace.csv', num_triplets=30000, training_triplets_path='datasets/training_triplets_30000.npy', transform=transforms)##using Triplet
# train_data = SiameseCASIAWebFace('CASIA-maxpy-clean', 'CASIA_anno.txt',transforms)##using Contrastive
# train_data = CASIAWebFace('CASIA-maxpy-clean', 'CASIA_anno.txt',transforms)

lfw_data = LFW('./LFW/lfw_align_112', './LFW/pairs.txt', transforms)
val_data = None
dataloaders = {'train': DataLoader(train_data, batch_size=BATCH_SIZE,shuffle=True, **kwargs),
               'LFW': DataLoader(lfw_data, batch_size=BATCH_SIZE,shuffle=False, **kwargs) }

ckpt_tag = 'se_resnet18'

# Set model
model = ResNet18()
# model = resnet18(pretrained=False,loss_fn='triplet',num_classes=256 )
summary(model.cuda(), (3, 112, 112))
model = model.to(device)

# print(margin_type)
margin=margin_type
# '''
# Set margin
if margin_type == 'Softmax':
    margin = nn.Linear(feature_dim, train_data.num_class)
elif margin_type == 'NormFace':
    margin = NormFace(feature_dim, train_data.num_class)
elif margin_type == 'SphereFace':
    margin = SphereFace(feature_dim, train_data.num_class)
elif margin_type == 'CosFace':
    margin = CosFace(feature_dim, train_data.num_class)
elif margin_type == 'ArcFace':
    margin = ArcFace(feature_dim, train_data.num_class)
else:
    raise NameError("Margin Not Supported!")
margin = margin.to(device)


#Set criterion
if criterion=='Softmax' or criterion=='NormFace' or criterion=='SphereFace'or criterion=='CosFace'or criterion=='ArcFace':
    criterion = torch.nn.CrossEntropyLoss().to(device)
elif criterion=='OHEM':
    criterion = OHEMCrossEntropyLoss(0.75).to(device)
elif criterion=='FocalLoss':
    criterion = FocalCrossEntropyLoss(3, 0.5).to(device)
elif criterion=='Contrastive':
    criterion = ContrastiveLoss(margin=0.8).to(device)
elif criterion=='Triplet':
    criterion = None#torch.nn.CrossEntropyLoss().to(device)
    # criterion = torch.nn.NLLLoss().to(device)


# if criterion=='Softmax' or criterion=='NormFace' or criterion=='SphereFace'or criterion=='CosFace'or criterion=='ArcFace':
#     print('ok')
#     if margin_type == 'Softmax'or margin_type == 'NormFace'or margin_type == 'SphereFace'or margin_type == 'CosFace'or margin_type == 'ArcFace':
#        from trainer import Trainer
#        print(margin_type,criterion)
#     else:
#         pass
# elif criterion=='Contrastive':
#     from Contrastive_trainer import Trainer
#     print(margin_type, criterion)
# elif criterion=='Triplet':
#     from Triplet_trainer import Trainer
#     print(margin_type, criterion)


# Set optimizer

optimizer = torch.optim.Adam([{'params': model.parameters()},
                              {'params': margin.parameters()}], lr=LR)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=STEPS, gamma=0.1)

# Set trainer
trainer =Trainer(EPOCHS, dataloaders, model, optimizer, scheduler, device,
                  margin, ckpt_tag,criterion)
trainer.train()







