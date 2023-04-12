#!/usr/bin/python                                                       
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn

import numpy as np
import cv2
import PIL.Image as Image

from .network_unet import *

colors_list = [
[0,0,255],
[0,255,0],
[255,0,0],
[0,255,255],
[255,0,255],
[255,255,0],
[0,0,128],
[128,128,255],
[255,123,123],
[0,102,255],
[102,153,51],
[153,255,255],
[204,204,51],
[0,204,255],
[150,150,150],
[255, 255, 255],
[0,0,0]
]

def get_bgr_image(image, label, out):
    ret_list = []
								    
    mean = [.485, .456, .406]                                       
    std = [.229, .224, .225]                                        
    mean = np.reshape(mean , [1, 1, -1])                            
    std = np.reshape(std, [1, 1, -1])                               
    img = image.data.cpu().numpy()                                  
    for idx in range(img.shape[0]):
        
        image = (img[idx, :3, :, :])
        image = image.transpose((1,2,0))
        image = image * std + mean
        image = image*255
        ret_list.append(image)
        

    img = label.data.cpu().numpy()                                     
    for idx in range(img.shape[0]):
        
        label_write = img[idx, :, :]
        label_dump = np.zeros_like(image)
        for label_idx in np.unique(label_write):
            if label_idx == 0:
                continue
            label_dump[label_write == label_idx] = colors_list[(label_idx-1)%len(colors_list)]

        ret_list.append(label_dump)

    img = out.data.cpu().numpy()
    for idx in range(img.shape[0]):
        out_write = img[idx, :, :]
        out_dump = np.zeros_like(image)
        for label_idx in np.unique(out_write):
            if label_idx == 0:
                continue
            out_dump[out_write == label_idx] = colors_list[(label_idx-1)%len(colors_list)]
        ret_list.append(out_dump)

    return ret_list

class segment_and_get_center:
    def __init__(self, checkpoint_path):
        self.image_size = 512
        self.input_transform = Compose([Resize((self.image_size, self.image_size)),
            ToTensor(), Normalize([.485, .456, .406], [.229, .224, .225])])
        # loading checkpoint
        device = torch.device('cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.net = model()
        self.net.load_state_dict(checkpoint)

    def segment(self, image):
        self.net.eval()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        out_size = image.shape
        image = self.input_transform(Image.fromarray(image)).unsqueeze(0)

        output = self.net(image)
        output = torch.argmax(output, dim=1)

        ret_list = get_bgr_image(image, output, output)

        output = output.data.cpu().numpy()[0,:,:]
        mask = output
        x, y = out_size[1], out_size[0]
        mask = cv2.resize(mask, dsize=(x,y), interpolation=cv2.INTER_NEAREST)
        return mask, ret_list
    
    def get_center(self, mask):
        y, x = np.argwhere(mask == 1)[:,0], np.argwhere(mask == 1)[:,1]
        y, x = np.mean(y), np.mean(x)
        return x, y