#!/usr/bin/env python
# coding: utf-8
# Author: Siddhartha Gairola (t-sigai at microsoft dot com))

# python internals
import math

# external libs
import numpy as np
import cv2                                          
import matplotlib.pyplot as plt
np.seterr('raise')

def arc_step(nrings, p, oz, oy, k):

	rocs, zs, ys = [], [], []

	for i in range(0, nrings+1):
		step = 0.01
		if i < 2:
			z_old = 0
			y_old = 0
			slope_old = 0
			z = 0
		if i > 0:
			z = z_old + slope_old*(y-y_old) + 0.5*quad_old*((y-y_old)**2)

		checker_counter = 0
		while True:
			#print("checker_counter", checker_counter, "idx", i)
			y = (-p+z)*k[i]
			if i == 0:
				quad_old = 2*z / y**2
				cube = 0.0
			if i > 0:
				cube = 6*(z- z_old - slope_old*(y-y_old) - 
					0.5*quad_old*((y-y_old)**2)) / (y-y_old)**3

			slope = slope_old + quad_old*(y-y_old)+0.5*cube*(y-y_old)**2
			k_obj = (oy[i] - y) / (-oz[i] + z)
			cos_o = (k_obj - slope) / math.sqrt((1 + k_obj**2)*(1+slope**2))
			cos_p = (k[i] + slope) / math.sqrt((1 + k[i]**2)*(1+slope**2))

			if (cos_o-cos_p)*step < 0:
				step = -step/3.0
			z += step
			checker_counter += 1

			# cut off condition
			if abs(step) <= 1e-7 or checker_counter > 1e4:
				break

		quad_old = quad_old + cube*(y-y_old)
		z_old = z
		y_old = y
		slope_old = slope
		zs.append(z)
		ys.append(y)

		curv = abs(quad_old) / (1 + slope**2)**1.5 # curvature
		roc = 1/curv # radius of curvature
		rocs.append(roc)

	#plt.figure()
	#plt.plot(zs, ys, c='r', ls='-')
	#plt.savefig('out/arc_out.png')
	#plt.close()
	return rocs, zs, ys