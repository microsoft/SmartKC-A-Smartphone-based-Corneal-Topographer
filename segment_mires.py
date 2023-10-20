import numpy as np
import cv2

from get_center.get_center import segment_and_get_center
from enhance_img.main_enhancement import enhance, detect_circle
from constants import Constants

class mire_segmentation:
    def __init__(self, seg_method, center, dl_seg_file = None):
        if seg_method == "dl":
            assert dl_seg_file is not None, "Please provide a weights file for the DL model"
            self.dl_seg_model = segment_and_get_center(dl_seg_file)
        self.seg_method = seg_method
        self.center = center
        
    def auto_canny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        # return the edged image
        return edged
    
    def remove_innermost_circle(self, image, center, filter_radius = 10):
        filter_mask = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
        filter_mask = cv2.circle(filter_mask, center, filter_radius, (255, 255, 255), -1)[:,:,0]
        filter_mask[filter_mask > 0] = 1
        image = image * (1 - filter_mask)
        return image

    
    def mire_seg_img_proc(self, image):
        image_clean, _ = enhance(image, downsample=Constants.IMG_PROC_SEG_PARAMS["DOWNSAMPLE"], blur=Constants.IMG_PROC_SEG_PARAMS["BLUR"])
        image_clean_inv = 255 - image_clean
        # edge computation        
        image_edge_orig, image_edge_inv = self.auto_canny(image_clean), self.auto_canny(image_clean_inv)
        # remove inner_most circle
        image_edge_inv = self.remove_innermost_circle(image_edge_inv, self.center, filter_radius = 10)
        image_clean_inv = self.remove_innermost_circle(image_clean_inv, self.center, filter_radius = 10)

        return image_clean_inv, image_edge_inv
    
    def mire_seg_dl(self, image):
        image = np.dstack((image, np.dstack((image, image))))
        mask, _ = self.dl_seg_model.segment(image)
        mask = mask.astype(np.uint8)
        return mask       

    def segment_mires(self, image):
        if self.seg_method == "img_proc":
            return self.mire_seg_img_proc(image)
        else:
            return self.mire_seg_dl(image), None