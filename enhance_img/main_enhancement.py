# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:42:58 2016

@author: utkarsh
"""

import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

from .image_enhance import image_enhance

def enhance(img, downsample=False, blur=True):

    if(len(img.shape)>2):
         img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    if blur == True:
        #img = cv2.GaussianBlur(img, (3,3),0)
        img = cv2.GaussianBlur(img, (5, 5), 10)
        #img = cv2.GaussianBlur(img, (7,7),10)
        #img = cv2.GaussianBlur(img, (15,15),10)
    rows,cols = np.shape(img)
    aspect_ratio = np.double(rows)/np.double(cols)
    new_rows, new_cols = rows, cols

    if downsample == True:
        new_rows = 350
        new_cols = new_rows/aspect_ratio

    img = cv2.resize(img,(np.int(new_cols),np.int(new_rows)))

    enhanced_img = image_enhance(img)
    enhanced_img = enhanced_img.astype(np.uint8)
    enhanced_img = cv2.resize(enhanced_img, (cols, rows), interpolation=cv2.INTER_LINEAR);
    enhanced_img = 255*enhanced_img

    edge_canny = cv2.Canny(enhanced_img,30,55)
    
    kernel = np.ones((5,5), np.uint8)                                           
    img_erosion = cv2.erode(enhanced_img, kernel, iterations=1)
    edge = enhanced_img - img_erosion                             
										
    return enhanced_img, edge

# This part of the code is not from the original pipeline
def detect_circle(img, ups=1):#Detecting hough circles in the gray scale image.
        # gray_blurred = cv2.blur(img, (3, 3))                          
	
        orig = img.copy()
        pt0=0                                                           
        pt1=0                                                           
        #detected_circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.0, 15, param1 = 55, param2 = 25, minRadius = 10*ups, maxRadius = 20*ups)
        detected_circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2.0, 600, minRadius = 100*ups, maxRadius = 150*ups)
        if detected_circles is not None:                                
                # Convert the circle parameters a, b and r to integers. 
                detected_circles = np.uint16(np.around(detected_circles))
                                                                        
                for pt in detected_circles[0, :]:                       
                        a, b, r = pt[0], pt[1], pt[2]                   
                        pt0 = a                                         
                        pt1 = b                                         
                        print('Center detected:', a,b)                                               
                        # Draw the circumference of the circle.         
                        cv2.circle(orig, (a, b), r, (100, 255, 0), 2)   
                        break                                           
        #plt.figure()                                                    
        #plt.imshow(orig)                                                
        #plt.show()                                                      
        return pt0,pt1