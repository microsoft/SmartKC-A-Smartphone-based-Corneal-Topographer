import os
import shutil
import cv2
import random
import numpy as np

dir = './19_06_2023/'
out_dir = './anurag_cropped_no_annotation/'

def process(dir, subdir, file):
    print(subdir, file)
    try:
        filename = os.path.join(subdir, file)
        image = cv2.imread(filename)
        image = image[85:420, 85:420,:]
        image = cv2.resize(image, (306,306), interpolation=cv2.INTER_LINEAR)

        # image = image[:306, :306, :]
        target_file = ('smartkc_' + file.split('_')[0]) + '_'
        target_file += ('OS' if 'left' in file else 'OD') + '_'
        target_file += 'curv' if 'tan' in file else 'axial'
        target_file += '.png'
        
        target_filpath = os.path.join(out_dir, target_file)
        
        cv2.imwrite(target_filpath, image)
    except Exception as e:
        print(e)
        print("ERROR")

for subdir, dirs, files in os.walk(dir):
    for file in files:
        if 'axial' in file or 'tan' in file:
            process(dir, subdir, file)