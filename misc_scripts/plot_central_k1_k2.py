import cv2
import math
import numpy as np

'''
f = open('../data/run_3_images.txt')
for idx, line in enumerate(f):
	line = line.strip().split(' ')
	print(idx, line[0]+'.jpg')
f.close()
exit()
'''

f = open('../run_3_output_scores_filtered.txt')
our_dict = {}
last = -1
for line in f:
	line = line.strip().split(' ')
	if len(line) == 2 and line[0] != "KISA":
		our_dict[line[1].split('.jpg')[0]] = []
		last = line[1].split('.jpg')[0]
	elif len(line) == 3:
		our_dict[last].append(float(line[1]))

	if len(line) == 14:
		our_dict[last].append(float(line[7]))
		our_dict[last].append(float(line[9]))
f.close()

base_dir = '../run_3_images_output/'
for idx in our_dict:
	print(idx, our_dict[idx])
	img_axial = cv2.imread(base_dir+idx+'_axial.png')
	img_tan = cv2.imread(base_dir+idx+'_tan.png')
	center = (img_axial.shape[1]//2, img_axial.shape[0]//2)
	angle = -our_dict[idx][0]*np.pi/180
	radii = [30]
	# drawing the angle
	x2, y2 =  int(center[0] - radii[0] * math.cos(angle)) , int(center[1] - radii[0] * math.sin(angle))
	cv2.putText(img_axial, str(our_dict[idx][1]), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255),1)
	cv2.putText(img_tan, str(our_dict[idx][1]), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255),1)

	# drawing perpendicular
	angle = angle + np.pi/2
	x1, y1 =  int(center[0] + radii[0] * math.cos(angle)) , int(center[1] + radii[0] * math.sin(angle))
	x2, y2 =  int(center[0] - radii[0] * math.cos(angle)) , int(center[1] - radii[0] * math.sin(angle))
	if x1 < x2:
		x2 , y2 = x1, y1
	cv2.putText(img_axial, str(our_dict[idx][2]), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255),1)
	cv2.putText(img_tan, str(our_dict[idx][2]), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255),1)

	cv2.imwrite("out/"+idx+'_axial.png', img_axial)
	cv2.imwrite("out/"+idx+'_tan.png', img_tan)