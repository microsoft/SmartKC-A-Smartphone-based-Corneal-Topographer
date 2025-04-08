import numpy as np
import cv2
import os
from colors import colors_list
from mire_detection import *
from utils import *


base_dir = "../data/image_masks_clean/"
image_names = os.listdir(base_dir)
image_names = [x for x in image_names if "mask" in x]
global yoo_count
yoo_count = 0
def func(name):
	# try:
		if not name.endswith('.jpg'):
			return
		base_dir = "../data/image_masks_clean/"
		out_dir = "../data/components/"
		n_mires = 20
		center = (250, 250)
		global yoo_count
		img = cv2.imread(base_dir+name)
		img[img < 128] = 0
		img[img>128] = 255
		img = 255-img
		output = cv2.connectedComponentsWithStats(
			img[:,:,0], 4, cv2.CV_32S)
		(numLabels, labels, stats, centroids) = output
		image_cent_list, center, others = process(img.copy()[:,:,0], img.copy()[:,:,0], 
			center, 1, 0, 360)
		r_pixels, coords = clean_points(image_cent_list, out_dir+'plot_'+name,
			center, n_mires, 1, 0, 360)

		img_out = np.zeros_like(img)
		img_gt = np.zeros_like(img[:,:,0])
		total_found = []
		for i in range(1, numLabels):
			#img_out[labels == i] = colors_list[i%len(colors_list)]
			curr_comp = np.argwhere(labels == i)
			mire_counts = []
			for j in range(len(coords)):
				mire_counts.append(0)

			for j in range(curr_comp.shape[0]):
				x = tuple(curr_comp[j])
				for k in range(len(coords)):
					if x in coords[k]:
						mire_counts[k] += 1

			if np.sum(mire_counts) == 0:
				continue
			curr_label = np.argmax(mire_counts)
			total_found.append(curr_label)
			img_out[labels == i] = colors_list[curr_label%len(colors_list)]
			img_gt[labels == i] = curr_label + 1

		
		img = cv2.putText(img, 'No. of Comps: '+str(len(np.unique(total_found))), (40, img.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
		img_out = np.concatenate([img, img_out], axis=1)
		cv2.imwrite(out_dir+"component_"+name, img_out)
		np.save(out_dir+"gtmask_"+name.strip().split('.')[0], img_gt)
		yoo_count += 1
		print("Done", yoo_count)
		return 
		#cv2.imwrite(out_dir+name, img)
		#print("Done ", idx, len(np.unique(total_found)))
		#break
	# except Exception as e:
	# 	print(f"Error processing {name}: {e}")
	# 	return
import multiprocessing
from multiprocessing import Pool
num_cores = multiprocessing.cpu_count()

if __name__ == "__main__":
    print("STARTED")
    pool = Pool()
    pool.map(func, image_names)
    print("COMPLETED")
    pool.close()
