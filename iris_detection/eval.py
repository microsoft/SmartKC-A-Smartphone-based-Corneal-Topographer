#!/usr/bin/python                                                       
# This has been adapted from https://github.com/sairin1202/Semantic-Aware-Attention-Based-Deep-Object-Co-segmentation/
# Please refer to the above link for more details or queries.           
                                                                        
import torch
import torchvision
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, ToTensor, ToPILImage
import torch.nn as nn
import skimage.io as io

import numpy as np
import cv2
import itertools
import PIL.Image as Image
import argparse
import os

from utils import *
from .network_unet import *

from tqdm import tqdm

# let the label pixels =1 if it >0
class Relabel:
    def __call__(self, tensor):
        assert isinstance(
            tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor > 0] = 1
        return tensor

# numpy -> tensor
class ToLabel:
    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long()

class iris_segmenter:
    def __init__(self):
        self.image_size = 512
        self.input_transform = Compose([Resize((self.image_size, self.image_size)), 
            ToTensor(), Normalize([.485, .456, .406], [.229, .224, .225])])
        self.label_transform = None

        # loading checkpoint
        device = torch.device('cpu')
        checkpoint = torch.load('./iris_detection/models_out/best_epoch_97_iter_0.pkl', 
            map_location=device)
        self.net = model()
        self.net.load_state_dict(checkpoint)

    def segment(self, image):

        self.net.eval()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out_size = image.shape
        image = self.input_transform(Image.fromarray(image)).unsqueeze(0)
        output = self.net(image)
        output = torch.argmax(output, dim=1)
        output = output.data.cpu().numpy()[0,:,:]
        out_dump = output
        x, y = out_size[1], out_size[0]
        out_dump = cv2.resize(out_dump, dsize=(x,y), interpolation=cv2.INTER_NEAREST)
        return out_dump
