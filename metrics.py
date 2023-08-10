#!/usr/bin/env python
# Author: Siddhartha Gairola (t-sigai at microsoft dot com))
import math
import logging
import numpy as np
import cv2
from utils import plot_line

def compute_simk(curv_map, center, cutoff_r):

	filter_mask = np.zeros((curv_map.shape[0], curv_map.shape[1], 3)).astype(np.uint8)
	filter_mask = cv2.circle(filter_mask, center, cutoff_r, (255, 255, 255), -1)[:,:,0]
	filter_mask[filter_mask>0] = 1

	mean_curv_list = []
	for angle in range(0, 180):
		curr_meridian = np.zeros((curv_map.shape[0], curv_map.shape[1], 3)) 
		curr_meridian_1 = plot_line(curr_meridian, center, angle)
		curr_meridian_2 = plot_line(curr_meridian, center, angle+180)
		curr_meridian = curr_meridian_1 + curr_meridian_2
		curr_meridian = curr_meridian[:,:,0]
		curr_meridian = curr_meridian*filter_mask
		mean_curv_list.append((curv_map[curr_meridian>0]).mean()*337.5)

	min_idx = np.argmin(mean_curv_list) # simK2 angle
	max_idx = (min_idx+90)%len(mean_curv_list) # simK1 angle
	logging.info("min_idx: {}, max_idx: {}".format(min_idx, max_idx))
	logging.info("SimK {}, {}".format(mean_curv_list[min_idx], mean_curv_list[max_idx]))
	return round(mean_curv_list[min_idx],2), round(mean_curv_list[max_idx],2), min_idx, max_idx

def compute_tilt_factor(curr_map, act_map, r1, r2, center, angle, image_name, output_folder="out"):
	mask = np.zeros((curr_map.shape[0], curr_map.shape[1], 3)).astype(np.uint8)
	# drawing the angle
	x1, y1 =  int(center[0] + r2 * math.cos(angle)) , int(center[1] + r2 * math.sin(angle)) 
	x2, y2 =  int(center[0] - r2 * math.cos(angle)) , int(center[1] - r2 * math.sin(angle))
	mask = cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 1)[:,:,0]
	central_ids = np.argwhere(mask>0)
	mask = mask > 0

	min_steep, max_steep = 1e9, -1
	p_min, p_max = (-1, -1), (-1, -1)

	# compute steepest area weight 1 mm diameter circle
	for idx in range(central_ids.shape[0]):
		curr_y, curr_x = central_ids[idx][0], central_ids[idx][1]
		curr_mask = np.zeros((curr_map.shape[0], curr_map.shape[1], 3)).astype(np.uint8)
		curr_mask = cv2.circle(curr_mask, (curr_x, curr_y), r1, (255, 255, 255), -1)[:,:,0]
		curr_mask_in = mask * (curr_mask > 0)
		curr_steep = curr_map[curr_mask_in].mean()
		if curr_steep > max_steep:
			max_steep = curr_steep
			p_max = (curr_x, curr_y)

		if curr_steep < min_steep:
			min_steep = curr_steep
			p_min = (curr_x, curr_y)

	act_map = cv2.circle(act_map, p_max, r1, (0,255,0), 2)
	act_map = cv2.circle(act_map, p_min, r1, (0,0,255), 2)
	logging.info("{} max_steep: {}, min_stepp: {}".format(image_name, max_steep, min_steep))
	#cv2.imwrite(output_folder+"/" + image_name + "/" + image_name + '_tilt_factor.png', act_map)
	return max_steep, min_steep

def clmi_ppk(curv_map, axial_map, r_2, r_8, center):
	mask = np.zeros((curv_map.shape[0], curv_map.shape[1], 3)).astype(np.uint8)
	mask = cv2.circle(mask, center, r_8, (255, 255, 255), -1)[:,:,0]
	central_ids = np.argwhere(mask>0)
	mask = mask > 0

	max_steep = -1
	p1 = (-1, -1)
	m1 = -2

	# compute steepest area weight 2 mm diameter circle
	for idx in range(central_ids.shape[0]):
		curr_y, curr_x = central_ids[idx][0], central_ids[idx][1]
		curr_mask = np.zeros((curv_map.shape[0], curv_map.shape[1], 3)).astype(np.uint8)
		curr_mask = cv2.circle(curr_mask, (curr_x, curr_y), r_2, (255, 255, 255), -1)[:,:,0]
		curr_mask_in = mask * (curr_mask > 0)
		curr_steep = curv_map[curr_mask_in].mean()
		if curr_steep > max_steep:
			max_steep = curr_steep
			p1 = (curr_x, curr_y)

			# points outside c1
			curr_mask_out = mask*(curr_mask <= 0)
			curr_steep_out = curv_map[curr_mask_out].mean()
			m1 = curr_steep - curr_steep_out

	p1_angle = np.arctan2((p1[1]-center[1]),(p1[0]-center[0])+1e-9)*180/np.pi
	p2_angle = p1_angle+180
	p1_mag = ((p1[0]-center[0])**2 + (p1[1]-center[1])**2)**0.5
	p2 = (int(center[0] + p1_mag * math.cos(float(p2_angle) * np.pi / 180.0)),
		int(center[1] + p1_mag * math.sin(float(p2_angle) * np.pi / 180.0)))

	# computing m2 for c2
	curr_mask = np.zeros((curv_map.shape[0], curv_map.shape[1], 3)).astype(np.uint8)
	curr_mask = cv2.circle(curr_mask, p2, r_2, (255, 255, 255), -1)[:,:,0]
	curr_mask_in = mask*(curr_mask>0)
	curr_steep = curv_map[curr_mask_in].mean()
	curr_mask_out = mask*(curr_mask<=0)
	curr_steep_out = curv_map[curr_mask_out].mean()
	m2 = curr_steep - curr_steep_out

	# computing clmi
	clmi = -1
	if p1_mag < r_2:
		clmi = m1 - p1_mag/r_2*m2
	else:
		clmi = m1 - m2

	clmi = clmi*337.5
	print(clmi, "clmi")
	ppk = np.exp(-6.4483+2.1319*clmi)/(1+np.exp(-6.4483+2.1319*clmi))
	ppk, clmi = round(ppk, 2), round(clmi, 2) # rounding off
	logging.info("PPK: {} CLMI: {} M1: {} M2: {}".format(ppk, clmi, m1, m2))

	axial_map = cv2.circle(axial_map, p1, r_2, (0,255,0), 2)
	axial_map = cv2.circle(axial_map, p2, r_2, (0,0,255), 2)
	#print("angles", p1_angle, p2_angle, "p1", p1, "p2", p2)
	#cv2.imwrite('out/'+str(clmi)+'_ppk_test.png', axial_map)
	return ppk, clmi, m1, m2

def KISA(curv_map, center, coords, r_3, AST, start_angle=0, end_angle=360, jump=1):

	# computing central_k
	central_k_list = []
	for idx, angle in enumerate(np.arange(start_angle, end_angle, jump)):
		for mire in range(3):
			y, x = coords[mire][idx]
			y, x = y+center[1], x+center[0]
			central_k_list.append(curv_map[int(y), int(x)])

	central_k = 337.5*np.mean(central_k_list)
	central_k = max(1, central_k-47.2)

	# computing SRAX
	lower_half_angle, upper_half_angle = -1, -1
	max_lower, max_upper = -1, -1
	for idx, angle in enumerate(np.arange(start_angle, end_angle, jump)):
		for mire in (2,15):
			y,x = coords[mire][idx]
			y,x = y+center[1], x+center[0]
			curr_k = curv_map[int(y), int(x)]
			if 10 <= angle and angle <= 170 and max_lower < curr_k:
				# lower
				max_lower = curr_k
				lower_half_angle = angle
			elif 190 <= angle and angle <= 350 and max_upper < curr_k:
				# upper
				max_upper = curr_k
				upper_half_angle = angle

	smaller_angle = upper_half_angle-lower_half_angle
	smaller_angle = min(smaller_angle, 360-smaller_angle)
	SRAX = max(180-smaller_angle, 1)

	# I-S index
	I_list, S_list = [], []
	angles = [30, 60, 90, 120, 150, 210, 240, 270, 300, 300]
	for angle in angles:
		x = int(center[0] + r_3 * math.cos(float(angle) * np.pi / 180.0))
		y = int(center[1] + r_3 * math.sin(float(angle) * np.pi / 180.0))
		if angle < 180:
			S_list.append(curv_map[y,x])
		else:
			I_list.append(curv_map[y,x])

	I, S = np.mean(I_list), np.mean(S_list)
	I_S = max(337.5*abs(I-S),1)

	kisa = central_k*I_S*AST*SRAX*0.33
	kisa = round(kisa, 2)

	logging.info("K: {} SRAX: {} I-S: {} KISA: {}".format(central_k, SRAX, I_S, kisa))
	
	return kisa, central_k, SRAX, I_S