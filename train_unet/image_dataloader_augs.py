#!/usr/bin/python                                                       

# This has been adapted from https://github.com/sairin1202/Semantic-Aware-Attention-Based-Deep-Object-Co-segmentation/
# Please refer to the above link for more details or queries.           
import numpy as np
import os
from PIL import Image
# torch imports
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
# skimage imports
from skimage.filters import gaussian
from skimage.filters import unsharp_mask
import skimage.io as io
import glob
import random
import torch
from distort import get_aug
import matplotlib.pyplot as plt
import cv2

def get_images(filename):
    image_names = np.genfromtxt(filename, dtype=str)
    return image_names[:,0]

def load_image(file):
    return Image.open(file)

def dump_files(image1, label1, name):
    cv2.imwrite('dump/'+name+'_img.jpg', cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
    label = np.zeros_like(image1)
    label[label1 == 1] = [255,255,255]
    label[label1 == 2] = [0,255,255]
    label[label1 == 3] = [0,0,255]
    cv2.imwrite('dump/'+name+'_mask.jpg', label)

def transformer(image, mask):
    # convert to PIL Image
    image, mask = Image.fromarray(image), Image.fromarray(mask)

    # image and mask are PIL image object. 
    img_w, img_h = image.size
    
    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)
  
    # Random affine
    affine_param = transforms.RandomAffine.get_params(
        degrees = [-180, 180], translate = [0.4,0.4],  
        img_size = [img_w, img_h], scale_ranges = [0.6, 1.4], 
        shears = [2,2])
    image = TF.affine(image, 
                      affine_param[0], affine_param[1],
                      affine_param[2], affine_param[3])
    mask = TF.affine(mask, 
                     affine_param[0], affine_param[1],
                     affine_param[2], affine_param[3])

    image = np.array(image)
    mask = np.array(mask)
    return image, mask

def to_grayscale(image, isrgb=True):
    if isrgb:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_out = np.zeros((image.shape[0], image.shape[1], 3))
    image_out[:,:,0] = image
    image_out[:,:,1] = image
    image_out[:,:,2] = image
    image_out = image_out.astype(np.uint8)
    return image_out


class image_loader(Dataset):
    def __init__(self, data_dir, label_dir, traintxt, image_size, input_transform=None, label_transform=None, split_type="train"):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.input_transform = input_transform
        self.label_transform = label_transform
        self.traintxt = traintxt
        self.train_names = get_images(self.traintxt)
        self.image_size = image_size
        self.split_type = split_type

    def __getitem__(self, index):

        # image name
        imagename1 = os.path.join(self.data_dir, self.train_names[index].split('.')[0]+'_crop.jpg')
        
        # ground truth labels
        #labelname1 = os.path.join(self.label_dir, ""+self.train_names[index].split('.')[0]+"_masks.npy")
        labelname1 = os.path.join(self.label_dir, "gtmask_"+self.train_names[index].split('.')[0]+"_masks.npy")

        # loading images
        with open(imagename1, "rb") as f:
            image1 = np.array(load_image(f).convert('RGB'))


        # loading labels
        with open(labelname1, "rb") as f:
            label1 = np.load(labelname1)

        if random.random() > 0.7 and self.split_type == "train":
            image1 = to_grayscale(image1)

        # distort image and label to do augmentations
        distort_prob = random.random()
        if distort_prob >= 0.5 and self.split_type == "train":
            image1 = get_aug(np.copy(image1), np.copy(label1))
            image1, label1 = transformer(image1, label1)
        
        #dump_files(image1, label1, self.train_names[index])

        if self.input_transform is not None:
            image1 = self.input_transform(Image.fromarray(image1))

        if self.label_transform is not None:
            label1 = self.label_transform(Image.fromarray(label1))
        
        # resizing labels
        label1 = cv2.resize(label1, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        
        return image1, torch.from_numpy(np.array(label1)).long() 

    def __len__(self):
        return len(self.train_names)
