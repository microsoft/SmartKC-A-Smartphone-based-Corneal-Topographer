#!/usr/bin/python                                                       

# This has been adapted from https://github.com/sairin1202/Semantic-Aware-Attention-Based-Deep-Object-Co-segmentation/
# Please refer to the above link for more details or queries.           
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import skimage.io as io
import glob
import random
import torch
import matplotlib.pyplot as plt
import cv2

def get_images(filename):
    image_names = np.genfromtxt(filename, dtype=str)
    return image_names

def load_image(file):
    return Image.open(file)

class image_loader(Dataset):
    def __init__(self, image, mask,image_size, input_transform=None, label_transform=None):
        self.image = image
        self.input_transform = input_transform
        self.label_transform = label_transform
        self.train_names = []
        images = os.listdir(self.data_dir)
        for image in images:
            image = image.strip().split('.')[0]
            self.train_names.append(image)
        self.image_size = image_size

    def __getitem__(self, index):

        # image name
        imagename1 = os.path.join(self.data_dir, self.train_names[index]+'.jpg')
        
        # loading images
        with open(imagename1, "rb") as f:
            image1 = np.array(load_image(f).convert('RGB'))

        out_size = image1.shape
        #dump_files(image1, label1, self.train_names[index])

        if self.input_transform is not None:
            image1 = self.input_transform(Image.fromarray(image1))

        return image1, self.train_names[index], out_size

    def __len__(self):
        return len(self.train_names)
