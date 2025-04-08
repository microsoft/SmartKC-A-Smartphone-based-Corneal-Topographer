#!/usr/bin/python                                                       

#importing stuff
import torch
import numpy as np
import os
from PIL import Image
import skimage.io as io
import random
import math
from math import sin, cos, pi
import matplotlib.pyplot as plt

#image augmentation
import imageio
# import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage

def get_aug(image, gt):

    # Randomly initialize flag1, flag2 & flag3 which add distortions to the input image.
    flag1, flag2, flag3 = random.randint(0, 5), random.randint(0, 4), random.randint(1,2)
    
    # Add Noise
    if flag1 == 0:    
        # Adds white/gaussian noise pixelwise to an image. The noise comes from the normal distribution N(L,S). 
        # If PCH is true, then the sampled values may be different per channel (and pixel).
        seq1 = iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))
    elif flag1 == 1:
        # Adds noise sampled from a laplace distribution following Laplace(L, S) to images. 
        # If PCH is true, then the sampled values may be different per channel (and pixel).
        seq1 = iaa.AdditiveLaplaceNoise(scale=(0, 0.1*255))
    elif flag1 == 2:
        # Adds noise sampled from a laplace distribution following Laplace(L, S) to images. 
        # If PCH is true, then the sampled values may be different per channel (and pixel).
        seq1 = iaa.AdditivePoissonNoise(lam=(10.0, 20.0))
    elif flag1 == 3:
        # Similar to SaltAndPepper, but only replaces with very white colors, i.e. no black colors.
        seq1 = iaa.Salt((0.0, 0.2))
    elif flag1 == 4:
        # Similar to SaltAndPepper, but only replaces with very black colors, i.e. no white colors.
        seq1 = iaa.Pepper((0.0, 0.2))
    elif flag1 == 5:
        # Replaces P percent of all pixels with very white or black colors. 
        # If PCH is true, then different pixels will be replaced per channel.
        seq1 = iaa.SaltAndPepper((0.0, 0.2))
    
    # Invert image or vary contrast
    if flag2 == 0:
        # Blurs images using a gaussian kernel with size S
        seq2 = iaa.GaussianBlur(sigma=(0.0, 3.0))
    elif flag2 == 1:
        # Applies gamma contrast adjustment following I_ij' = I_ij**G', where G' is a gamma value sampled from G and I_ij 
        # a pixel (converted to 0 to 1.0 space). If PCH is true, a different G' is sampled per image and channel.
        seq2 = iaa.GammaContrast(gamma=(0.5, 2.0))
    elif flag2 == 2:
        # Similar to GammaContrast, but applies I_ij = G' * log(1 + I_ij), where G' is a gain value sampled from G.
        seq2 = iaa.LogContrast(gain=(0.5, 1.0))
    elif flag2 == 3:
        # Similar to GammaContrast, but applies I_ij' = 1/(1 + exp(G' * (C' - I_ij))), where G' is a gain value sampled
        # from G and C' is a cutoff value sampled from C.
        seq2 = iaa.SigmoidContrast(gain=(5, 20), cutoff=(0.25, 0.75))
    elif flag2 == 4:
        # Similar to GammaContrast, but applies I_ij = 128 + S' * (I_ij - 128), where S' is a strength value sampled from S. 
        # This augmenter is identical to ContrastNormalization (which will be deprecated in the future).
        seq2 = iaa.LinearContrast(alpha=(0.0, 1.0))
    
    
        
    # Warp, Rotate, Scale, Translate or Flip
    if flag3 == 0:
        seq3 = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.02, 0.05))])
    elif flag3 == 1:
        seq3 = iaa.Sequential([iaa.Flipud(1)])
    elif flag3 == 2:
        seq3 = iaa.Sequential([iaa.Fliplr(1)])
    elif flag3 == 3:
        seq3 = iaa.Sequential([iaa.Affine(scale={"x": (0.5, 0.9), "y": (0.5, 0.9)})])
    elif flag3 == 4:
        seq3 = iaa.Sequential([iaa.Affine(rotate=(-135, 135))])
    elif flag3 == 5:
        seq3 = iaa.Sequential([iaa.ElasticTransformation(alpha=90, sigma=9)])
    elif flag3 == 6:
        seq3 = iaa.Sequential([iaa.Affine(translate_px={"x": (-40, 40), "y": (-40, 40)})])
        
    # Apply Transformation 1
    image = seq1.augment_image(image)

    # Apply Transformation 2
    image = seq2.augment_image(image)

    '''

    heat_map = np.zeros((gt.shape[0], gt.shape[1], 1)).astype(np.float32)
    
    heat_map[:,:,0] = gt
    
    # convert to heat map object
    heat_map =  HeatmapsOnImage(heat_map, shape=image.shape, min_value = 0, max_value = 3)
    
    image, heat_map = seq3(image=image, heatmaps=heat_map)
    heat_map = (heat_map.get_arr())

    gt = heat_map[:,:0]
    '''

    return image.astype(np.uint8)

