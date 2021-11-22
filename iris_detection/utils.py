#!/usr/bin/python                                                       
import torch
import torchvision

import glob
import cv2
import PIL.Image as Image
import os
import numpy as np
import cv2
import random

#skimage
from skimage import measure
from skimage import filters

###############IMAGE LOADER UTILITY FUNCTIONS##########################


def get_label(guidance, rough_mask):

    guidance = guidance.astype(np.float32)
    rough_mask = rough_mask.astype(np.float32)

    rough_mask[rough_mask > 128] = 255                                  
    rough_mask[rough_mask <= 128] = 0                                   
                                                                        
    guidance[guidance < 64] = 0                                         
    guidance[np.multiply(guidance >= 64, guidance < 144)] = -1          
    guidance[guidance >= 144] = 1                                       
                                                                        
    label = guidance*255 + rough_mask                                   
                                                                        
    return label.astype(np.uint8) 

###############END IMAGE LOADER UTILITY FUNCTIONS######################


###############TRAIN UTILITY FUNCTIONS################################

def pixel_accuracy(output, label):                            
    correct = len(output[output == label])                          
    wrong = len(output[output != label])                            
    return correct, wrong                                           
								    
def jaccard(output, label):                                   
    temp = output[label == 1]                                       
    i = len(temp[temp == 1])                                        
    temp = output + label                                           
    u = len(temp[temp > 0])                                         
    return i, u                                                     
								    
def precision(output, label):                                 
    temp = output[label == 1]                                       
    tp = len(temp[temp == 1])                                       
    p = len(output[output > 0])                                     
    return tp, p                                                    
								    
def save_inter(image, label, out, epoch_iteration, iteration, flag, output_dir):
								    
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
        
        cv2.imwrite(os.path.join(output_dir, epoch_iteration)+'/'
                +epoch_iteration+'_'+str(iteration)+'_'+str(idx)+'_'+str(flag)+'_image.png',
                cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_RGB2BGR))

    img = label.data.cpu().numpy()                                     
    for idx in range(img.shape[0]):
        
        label_write = img[idx, :, :]
        label_dump = np.zeros_like(image)
        label_dump[label_write == 1] = [0,0,255]
        label_dump[label_write == 2] = [0,255,255]
        label_dump[label_write == 3] = [255,255,255]
        
        cv2.imwrite(os.path.join(output_dir, epoch_iteration)+'/'
		+epoch_iteration+'_'+str(iteration)+'_'+str(idx)+'_'+str(flag)+'_gt.png',label_dump)

    img = out.data.cpu().numpy()
    for idx in range(img.shape[0]):
        out_write = img[idx, :, :]
        out_dump = np.zeros_like(image)
        out_dump[out_write == 1] = [0,0,255]
        out_dump[out_write == 2] = [0,255,255]
        out_dump[out_write == 3] = [255, 255, 255]
        cv2.imwrite(os.path.join(output_dir, epoch_iteration)+'/'
		+epoch_iteration+'_'+str(iteration)+'_'+str(idx)+'_'+str(flag)+'_pred.png',out_dump)
        
								    
def save_stats(epoch, iteration, value, flag, output_dir):    
								    
    arr = np.asarray([epoch, iteration]+value)                      
    np.savetxt(os.path.join(output_dir, flag+str(epoch)+'_'+str(iteration)+'.txt'), arr)
    print(flag+" stats saved") 

def guidance_weights(guide, scale):
    
    weights = torch.where(guide != 0, torch.tensor(scale), torch.tensor(1))

    return weights.float()



###############END TRAIN UTILITY FUNCTIONS############################
