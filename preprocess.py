#!/usr/bin/env python
# Author: Siddhartha Gairola (t-sigai at microsoft dot com)
import logging
import numpy as np
import cv2
import os

# external modules
from enhance_img.main_enhancement import enhance, detect_circle
from iris_detection.eval import iris_segmenter
from get_center.get_center import segment_and_get_center
# from crf import DenseCRF

def undo_zoom(image, zoom_factor):
    height, width = image.shape[:2]
    # 4x zoom
    image = cv2.resize(image, (width//zoom_factor, height//zoom_factor), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image.copy(),(height-height//zoom_factor)//2,(height-height//zoom_factor)//2,
        (width-width//zoom_factor)//2,(width-width//zoom_factor)//2,cv2.BORDER_CONSTANT,value=[0,0,0])
    return image, ((width-width//zoom_factor)//2, (height-height//zoom_factor)//2)


def mouse_click(event, x, y, flags, param):
    global centerX, centerY
    if event == cv2.EVENT_LBUTTONDOWN:
        centerX, centerY = x, y
        logging.info("Center selected: {}, {}".format(x,y))

def crop_around_center(image, crop_dims=(0,0), center=None, marked_center = None):
    if center is None:
        height, width = image.shape[:2]
        c_y_min, c_x_min = height // 2 - crop_dims[1] // 2, width // 2 - crop_dims[1] // 2
        c_y_max, c_x_max = height // 2 + crop_dims[1] // 2, width // 2 + crop_dims[0] // 2
        image_crop = image[c_y_min:c_y_max, c_x_min:c_x_max]
        # Adjust marked center because the image was cropped
        if (marked_center != None): marked_center = [marked_center[0]-c_x_min, marked_center[1]-c_y_min]
    else:
        centerX, centerY = center
        c_y_min, c_x_min = centerY - crop_dims[1] // 2, centerX - crop_dims[1] // 2
        c_y_max, c_x_max = centerY + crop_dims[1] // 2, centerX + crop_dims[1] // 2
        image_crop = image[c_y_min:c_y_max, c_x_min:c_x_max]
        # Adjust marked center because the image was cropped
        if (marked_center != None): marked_center = [marked_center[0]-c_x_min, marked_center[1]-c_y_min]

    return image_crop, marked_center


def preprocess_image(
    base_dir,
    image_name,
    center,
    crop_dims=(800, 800),
    iso_dims=500,
    output_folder="out",
    center_selection="manual",
    marked_center = None
):
    script_dir = os.path.dirname(__file__)
    # NOTE THIS EXPECTS THE INPUT IMAGE TO HAVE 3000x4000 RESOLUTION

    # Step 1: Image Centering and Cropping
    # Assuming image in always in the central region,
    # cropping an area of crop_dims
    image = cv2.imread(base_dir + "/" + image_name)
    height = image.shape[0]
    width = image.shape[1]
    image, offset = undo_zoom(image, 2) # take zoom automatically from response/app
    image_name = image_name.split(".jpg")[0]
    
    # Translate marked center to pixels
    if(marked_center):
        marked_center = [int(marked_center[0]*width)//2 + offset[0], int(marked_center[1]*height)//2 + offset[1]]

    # crop image to bring it to a reasonable size (1200 x 1200)
    image_crop = None
    marked_center_crop = None
    # if center[0] != -1 and center[1] != -1:
    #     center = (center[0]//2+offset[0], center[1]//2+offset[1]) # divide by zoom
    #     image_crop, marked_center_crop = crop_around_center(image, crop_dims=crop_dims, center=center, marked_center = marked_center)
    #     center = (image_crop.shape[1]//2, image_crop.shape[0]//2)
    # else:
    image_crop, marked_center_crop = crop_around_center(image, crop_dims=crop_dims, marked_center = marked_center)
    
    # Step 3: Locate Image Center
    if center[0] == -1 and center[1] == -1:

        if center_selection=="auto_obsolete":
            # this is for automatically detecting center
            image_gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
            image_clean, image_edge = enhance(image_gray, 
                downsample=downsample, blur=blur)
            image_edge_orig = auto_canny(image_clean)
            center = detect_circle(image_edge_orig, ups=1)
            
        elif center_selection=="auto":
            # this is detecting the center using UNet segmentor
            get_center_obj = segment_and_get_center(script_dir+'/get_center/segment_and_get_center_epoch_557_iter_14.pkl')
            image_temp_isocrop, _ = crop_around_center(image_crop, crop_dims=(iso_dims, iso_dims))
            mask, _ = get_center_obj.segment(image_temp_isocrop)
            #image_bgr, mask_bgr, _ = ret_list
            center = get_center_obj.get_center(mask)
            center = int(center[0])+(crop_dims[1]-iso_dims)//2, int(center[1])+(crop_dims[0]-iso_dims)//2
            #cv2.imwrite("image_bgr.png", image_bgr)
            #cv2.imwrite("mask_bgr.png", mask_bgr)
            
        elif center_selection=="manual-pc":
            # detect center manually
            cv2.imshow('image', image_crop)
            cv2.setMouseCallback('image', mouse_click)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            global centerX, centerY
            center = [centerX, centerY]
        elif center_selection == "manual-android":
            # draw marked center
            x = marked_center_crop[0]
            y = marked_center_crop[1]
            # debug: show marked center on image
            # temp = image_crop.copy()
            # cv2.line(temp, (x-10, y-10), (x+10, y+10), (255, 0, 0), 3)
            # cv2.line(temp, (x+10, y-10), (x-10, y+10), (255, 0, 0), 3)
            # cv2.imshow('image_after_temp', temp)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            center = [x, y]
            

    # isolate crop to only capture the central corneal region
    image_crop, _ = crop_around_center(image_crop, crop_dims=(iso_dims, iso_dims), center=center)

    # save original image
    cv2.imwrite(output_folder + "/" + image_name + "/" + image_name + "_out_col.png", image_crop)
    # save gray-scale image
    image_gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    center = (image_gray.shape[1] // 2, image_gray.shape[0] // 2)
    return image_gray, center, image_crop

    #cv2.imwrite(output_folder + "/" + image_name + "/" + image_name + "_out_gray.png", image_gray)
    #cv2.imwrite("./" + image_name + "_out_gray.png", image_gray)

    # Step 2: Image Enhancement, Cleaning & Enhancement
    # Processing input images to get clean thresholded images
    image_clean, image_edge = enhance(image_gray, downsample=downsample, blur=blur)

    # inverse image
    image_clean_inv = 255 - image_clean
    #cv2.imwrite(output_folder + "/" + image_name + "/" + image_name + "_out_clean_inv.png", image_clean_inv)
    #cv2.imwrite("./"+ image_name + "_out_clean_inv.png", image_clean_inv)

    # edge computation
    image_edge_orig, image_edge_inv = auto_canny(image_clean), auto_canny(image_clean_inv)
    #cv2.imwrite(output_folder + "/" + image_name + "/" + image_name + "_out_edge_inv.png", image_edge_inv)

    # get masked image
    image_mask = image_clean_inv.copy()
    image_mask[image_mask <= 128] = 0
    image_mask[image_mask > 128] = 1
    image_gray_masked = image_gray * image_mask
    #cv2.imwrite(output_folder + "/" + image_name + "/" + image_name + "_out_masked.png", image_gray_masked)
    
    # applying crf, function below
    # apply_crf(image_crop, image_mask.astype(np.float32))

    # Step 3: Locate Image Center


    # remove inner_most circle
    filter_mask = np.zeros_like(image_crop).astype(np.uint8)
    filter_mask = cv2.circle(filter_mask, center, filter_radius, (255, 255, 255), -1)[:,:,0]
    filter_mask[filter_mask > 0] = 1
    image_edge_inv = image_edge_inv * (1 - filter_mask)
    image_clean_inv = image_clean_inv * (1 - filter_mask)

    #cv2.imwrite(output_folder + "/" + image_name + "/" + image_name + "_out_edge_inv_filter.png", image_edge_inv)
    #cv2.imwrite(output_folder + "/" + image_name + "/" + image_name + "_out_clean_inv_filter.png", image_clean_inv)
    return image_gray, image_clean_inv, image_edge_inv, center