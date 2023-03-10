import json
from pathlib import Path

import os
import numpy as np
import torch
from PIL import Image

from panopticapi.utils import rgb2id


class BSIPanoptic:
    def __init__(self, img_folder, ann_folder, img_file, f_values, transforms=None, return_masks=True):
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.img_file = ['/'.join(img_file[aa][:-1].split('/')[-2:]) for aa in range(len(img_file))]
        self.feat_file = {kk[0].split('/')[1]:kk[1:len(kk)-1] for kk in [k.split(';') for k in [v.split('\n')[0] for v in f_values]]}
        self.transforms = transforms
        self.return_masks = return_masks

    def __getitem__(self, idx):
        img_file_info = self.img_file[idx]

        img = torch.load(self.img_folder+'inputs/'+(img_file_info.split('/'))[1]) 
        w, h = img.shape[2],img.shape[1]

        new_img = img_file_info.split('/')[1].split('.')[0] + '.jpg'
        

        target = {}
        target['image_id'] = torch.as_tensor([idx]) 
        if self.return_masks:
            target['masks'] = torch.load(self.img_folder+'instances/'+(img_file_info.split('/'))[1]) # masks
        target['labels'] = torch.load(self.img_folder+'classes/'+(img_file_info.split('/'))[1]) # labels

        target["boxes"] = torch.load(self.img_folder+'bounding_boxes/'+(img_file_info.split('/'))[1]) # masks_to_boxes(masks)

        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        
        #Features
        nm = img_file_info.split('/')[1][:-3]
        nm = nm.split('_')
        nm='_'.join(nm[:2]) + '.jpg'
        #print(idx, nm)
        target['granularity'] = torch.full(target['labels'].shape, torch.as_tensor([int(j.split(' ')[0])-2 for j in self.feat_file[nm]])[0], dtype=torch.int64)
        target['cyt_color']   = torch.full(target['labels'].shape, torch.as_tensor([int(j.split(' ')[1])-2 for j in self.feat_file[nm]])[0], dtype=torch.int64)
        target['nuc_shape']   = torch.full(target['labels'].shape, torch.as_tensor([int(j.split(' ')[2])-2 for j in self.feat_file[nm]])[0], dtype=torch.int64)
        target['swrbc']       = torch.full(target['labels'].shape, torch.as_tensor([int(j.split(' ')[3])-2 for j in self.feat_file[nm]])[0], dtype=torch.int64)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.img_file)

    def get_height_and_width(self, idx):
        img_file_info = self.img_file[idx]
        tmp = torch.load(self.img_folder+'inputs/'+(img_file_info.split('/'))[1])
        width, height = tmp.shape[1],tmp.shape[2]
        # height = img_info['height']
        # width = img_info['width']
        return height, width


def build(image_set, args):
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    # Our dataset
    train_img_folder_root = '../dataset/LeukoX/Fold_5/train/' 
    train_ann_folder_root = '../dataset/LeukoX/Fold_5/train/' 
    train_file = open('../dataset/LeukoX/Fold_5/train.txt','r')
    train_file_names = train_file.readlines()
    
    val_img_folder_root = '../dataset/LeukoX/Fold_5/val/'
    val_ann_folder_root = '../dataset/LeukoX/Fold_5/val/'
    val_file = open('../dataset/LeukoX/Fold_5/val.txt','r')
    val_file_names = val_file.readlines()

    #For testing only
    #val_img_folder_root = '../dataset/LeukoX/Fold_5/test/'
    #val_ann_folder_root = '../dataset/LeukoX/Fold_5/test/'
    #val_file = open('test_explain.txt','r')
    #val_file_names = val_file.readlines()
    
    feat_file = open('User1.txt','r') # File containing the rules
    feat_file_values = feat_file.readlines()

    
    #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    mode = 'panoptic'
    PATHS = {
        "train": (train_img_folder_root, train_ann_folder_root, train_file_names, feat_file_values),
        "val": (val_img_folder_root, val_ann_folder_root, val_file_names, feat_file_values),
    }

    img_folder, ann_folder, img_files, feature_values = PATHS[image_set]
    dataset = BSIPanoptic(img_folder, ann_folder, img_files, feature_values)
                          

    return dataset
