#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 09:59:11 2022

@author: ever
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CIFAR10C(Dataset):
    def __init__(self,labels_file,img_file,transform= None):
        self.image_label = np.load(labels_file)
        self.img = np.load(img_file)
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self,idx):
        img = Image.fromarray(np.uint8(self.img[idx])).convert('RGB')
        label= torch.tensor(int(self.image_label[idx]))

        if self.transform:
            img=self.transform(img)

        return (img,label)