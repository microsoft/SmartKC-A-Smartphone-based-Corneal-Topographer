import cv2
import numpy as np
from enhance_img.main_enhancement import enhance, detect_circle

def threshold_image(image):
	#image = cv2.GaussianBlur(image, (5,5), 0)
	#image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	#	cv2.THRESH_BINARY, 101, 5)
	_, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY
		+ cv2.THRESH_OTSU)
	return image

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def morph_enhance(image, kernel=None):

	image = image.astype(np.float32)
	if kernel == None:
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
		kernel[kernel > 0] = 255


	open_top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
	closed_top_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
	diff = closed_top_hat - open_top_hat
	diff = 255 - 255.0*(diff - diff.min())/(diff.max() - diff.min())
	edge = auto_canny(diff.astype(np.uint8))

	#cv2.imwrite('oth.png', open_top_hat)
	#cv2.imwrite('cth.png', closed_top_hat)
	cv2.imwrite('diff.png', diff.astype(np.uint8))
	#cv2.imwrite('canny.png', edge.astype(np.uint8))

	return diff.astype(np.uint8)



if __name__ == "__main__":

	'''
	for idx in range(17):
		#image_name = 'new_images_2/'+str(idx+1)
		image_name = 'images/pallavi'
		image = cv2.imread(image_name+'.jpg')
		height, width, _ = image.shape
		center = (height//2, width//2)
		crop = image[center[0]-400:center[0]+400, center[1]-400:center[1]+400, :]
		cv2.imwrite(image_name+'_crop.jpg', crop)

	exit()
	'''

	image_name = 'out/kt_calib_out_gray.png'
	image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2GRAY)
	edge_before = auto_canny(image)
	image = threshold_image(image)
	edge = auto_canny(image)
	cv2.imwrite('threshold_image.png', image)
	cv2.imwrite('edge_map.png', edge)
	cv2.imwrite('edge_map_before.png', edge_before)
	#image = morph_enhance(image)
	#enhanced_image, _ = enhance(image)
	#enhanced_image = 255 - enhanced_image
	#enhanced_image[enhanced_image > 0] = 1
	#image = np.multiply(image, enhanced_image)
	#cv2.imwrite('after_mult.png', image)

	#cv2.imwrite('gray_image.png', image)
	#cv2.imwrite('enhanced_image.png', enhanced_image)
	

