#!/usr/bin/python                                                       
                                                                        
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn
from .unet import UNet


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        # U-Net: Takes in the predicted mask and rough mask to produce the guidance.
        self.unet = UNet(in_channels = 3, n_classes = 4, wf = 5, padding=True, batch_norm=True, up_mode='upsample')


    def forward(self, x):
        # Takes input x: image, output out: segmentation map
        out = self.unet(x)
        return out
