import numpy as np
import cv2

def draw_dotted_line(image, center, width):
	for idx in range(50, width-50, 5):
		cv2.circle(image, (idx, center[1]), 1, (0, 0, 255), -1)

	return image


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	v = np.mean(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	print("Median", v, lower, upper, "Mean", np.mean(image))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def detect_iris(image, draw_image, center):
	rows, cols, _ = image.shape
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray_image = cv2.GaussianBlur(gray_image, (11, 11), 0)
	_, threshold = cv2.threshold(gray_image, 40, 255, cv2.THRESH_BINARY_INV)
	edge = auto_canny(threshold)
	mask = np.zeros_like(image)
	mask = cv2.line(mask, (0, center[1]), (cols, center[1]), (255, 255, 255), 1)[:,:,0]
	edge = (edge > 128) * (mask == 255)
	y,x = np.argwhere(edge==True)[:,0], np.argwhere(edge==True)[:,1]
	iris_radius = max(x)-center[0]
	cv2.circle(draw_image, center, iris_radius, (0, 255, 255), 2)
	draw_image = draw_dotted_line(draw_image, center, cols)
	#cv2.imshow("edge", edge)
	#cv2.imshow("iris", image)
	#cv2.waitKey(0)
	return draw_image

def detect_circles(image, minR, maxR):

	#image = cv2.medianBlur(image,3)
	edge = auto_canny(image)
	kernel = np.ones((5,5), np.uint8)
	edge = cv2.dilate(edge, kernel, iterations=2)
	cv2.imwrite("zoom/edge_image.png", edge)
	#cv2.imshow("edge", edge)
	#cv2.waitKey(0)

	#output = image.copy()
	output = np.zeros((image.shape[0], image.shape[1], 3))
	output[:,:,0] = edge
	output[:,:,1] = edge
	output[:,:,2] = edge
	if len(image.shape) > 2:
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	else:
		gray = image.copy()
	# detect circles in the image
	circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2.4, 1200, minRadius=minR, maxRadius=maxR) #1.2
	# ensure at least some circles were found
	x, y = image.shape[1]//2, image.shape[0]//2
	if circles is not None:
		#print("Circle found")
		# convert the (x, y) coordinates and radius of the circles to integers
		circles = np.round(circles[0, :]).astype("int")
		# loop over the (x, y) coordinates and radius of the circles
		for (x, y, r) in circles:
			# draw the circle in the output image, then draw a rectangle
			# corresponding to the center of the circle
			cv2.circle(image, (x, y), r, (0, 255, 0), 8)
			#cv2.rectangle(image, (x - 20, y - 20), (x + 20, y + 20), (0, 128, 255), -1)
			#cv2.line(image, (x - 40, y ), (x + 40, y), (0, 0, 255), 4)
			#cv2.line(image, (x, y - 40 ), (x, y+40), (0, 0, 255), 4)
		# show the output image
		#cv2.imshow("output", np.hstack([image, output]))
		#cv2.imshow("output", output)
		#cv2.waitKey(0)

	return image, [x,y]


base_dir = "../run_3_images/"
images_file_name = "../data/run_3_images.txt"
image_name_list = []
f = open(images_file_name)
for line in f:
	line = line.strip().split(' ')
	image_name_list.append(line[0])
f.close()


def crop_zoomed(image, zf):
	height, width, _ = image.shape
	height, width = height//zf, width//zf
	crop_dims = [width, height]
	height, width, _ = image.shape
	c_y_min, c_x_min = height//2-crop_dims[1]//2, width//2-crop_dims[0]//2
	c_y_max, c_x_max = height//2+crop_dims[1]//2, width//2+crop_dims[0]//2
	image_crop = image[c_y_min:c_y_max, c_x_min: c_x_max]
	return image_crop


base_dir = './circle/'
image_name_list = ["right_0"]
#image_name_list = ["left_0", "left_3", "right_2"]
coords = []
for idx, image_name in enumerate(image_name_list):
	#if idx == 7 or idx == 9:
	#	continue
	image = cv2.imread(base_dir+image_name+".jpg")
	height, width, _ = image.shape
	image = cv2.resize(image, (width//2, height//2), interpolation=cv2.INTER_LINEAR)
	image_small, center = detect_circles(image.copy(), 300, 800)
	#image_small, center = detect_circles(image.copy(), 1000, 1600)
	#cv2.imwrite("zoom/"+str(idx)+"_circle.png", image_small)
	cv2.imwrite("./circle/"+str(idx)+"_circle.png", image_small)
	print("Image", idx, center)
	coords.append(center)
	#print("image written")
	continue
	cv2.line(image, (width//2-50, height//2), (width//2+50, height//2), (0,0,255), 2)
	cv2.line(image, (width//2, height//2-50), (width//2, height//2+50), (0,0,255), 2)
	cv2.imwrite("zoom/"+str(idx)+"_"+str(1)+".png", image)
	for zf in range(2,5):
		image_crop = crop_zoomed(image.copy(), zf)
		image_crop = cv2.resize(image_crop, (width, height), interpolation=cv2.INTER_LINEAR)
		cv2.imwrite("zoom/"+str(idx)+"_"+str(zf)+".png", image_crop)
coords = np.array(coords)
print("x mean", coords[:,0].mean(), coords[:,0].std())
print("y mean", coords[:,1].mean(), coords[:,1].std())

img = np.zeros((4000, 3000, 3))
#img[int(coords[:,1].mean()), int(coords[:,0].mean()), :] = [0,0,255]
cv2.circle(img, (int(coords[:,0].mean()), int(coords[:,1].mean())), 4, (0,0,255), -1)
cv2.imwrite("zoom/test_0.png", img.astype(np.uint8))
img = crop_zoomed(img, 4)
img = cv2.resize(img, (3000, 4000), interpolation=cv2.INTER_LINEAR)
cv2.imwrite("zoom/test.png", img)

exit()

for idx, image_name in enumerate(image_name_list):
	print("Running", idx)
	image = cv2.imread(base_dir+image_name+".jpg")
	crop_dims = [800, 800]
	height, width = image.shape[:2]
	c_y_min, c_x_min = height//2-crop_dims[1]//2, width//2-crop_dims[0]//2
	c_y_max, c_x_max = height//2+crop_dims[1]//2, width//2+crop_dims[0]//2
	image_crop = image[c_y_min:c_y_max, c_x_min: c_x_max]
	image_small, center = detect_circles(image_crop.copy(), 5, 50)
	image_iris = detect_iris(image_crop.copy(), image_small, center)
	#cv2.imwrite("iris/"+str(idx)+"_small.png", image_small)
	cv2.imwrite("iris/"+str(idx)+"_iris.png", image_iris)