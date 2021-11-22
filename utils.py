 #!/usr/bin/env python
# Author: Siddhartha Gairola (t-sigai at microsoft dot com))
import os
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as __cm__
from matplotlib.ticker import LinearLocator as __LinearLocator__
from matplotlib.ticker import FormatStrFormatter as __FormatStrFormatter__


def check_angle(angle, skip_angles):
    # assuming angle lies between 0 and 360 degrees
    s1, e1 = skip_angles[0][0], skip_angles[0][1]
    s2, e2 = skip_angles[1][0], skip_angles[1][1]

    # no skip angles
    if s1 == -1 and e1 == -1 and s2 == -1 and e2 == -1:
        return 2

    # case 1: s1 and e1 are either side of 0
    if s1 > e1:
        if (s1 <= angle and angle <= 360) or (0 <= angle and angle <= e1):
            return 1
        if e1 < angle and angle < s2:
            return 2
        if s2 <= angle and angle <= e2:
            return 3
        if e2 < angle and angle < s1:
            return 4

    # case 2: 0 < s1, e1 < 180
    if e1 > s1 and e1 < 180:
        if s1 <= angle and angle <= e1:
            return 1
        if e1 < angle and angle < s2:
            return 2
        if s2 <= angle and angle <= e2:
            return 3
        if (e2 < angle and angle <=360) or (0 <= angle and angle < s1):
            return 4

    # case 3: 180 < s1, e1 < 360
    if e1 > s1 and e1 < 360:
        if s1 <= angle and angle <= e1:
            return 1
        if (e1 < angle and angle <= 360) or (0 <= angle and angle <s2):
            return 2
        if s2 <= angle and angle <= e2:
            return 3
        if e2 < angle and angle < s1:
            return 4

def get_dist(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def plot_color_rb(img, points):

    if len(points[0]) > 3:
        # for entire set of points (here points are (y,x))
        for mire in range(len(points)):
            for idx, point in enumerate(points[mire]):
                if mire%2 == 1:
                    img[int(point[0]), int(point[1]), :] = [0, 255, 255] # yellow
                else:
                    img[int(point[0]), int(point[1]), :] = [0, 0, 255] # red
    else:
        # for each meridian
        # initially points were (x,y)
        for idx, point in enumerate(points):
            if idx%2 == 0:
                img[int(point[1]), int(point[0]), :] = [0, 255, 255] # yellow
            else:
                img[int(point[1]), int(point[0]), :] = [0, 0, 255] # red

    return img

# Function to plot_line at an angle
def plot_line(img, center, angle):                                          
    img = np.zeros_like(img)                                                
    x_min, y_min = 0, 0                                                     
    x_max, y_max = img.shape[1], img.shape[0]                               
                                                                            
    # finding extreme for this line                                         
    length = 1                                                              
    allowed_len = -1                                                        
    while True:                                                             
        x =  int(center[0] + length * math.cos(angle * np.pi / 180.0))  
        y =  int(center[1] + length * math.sin(angle * np.pi / 180.0))  
        if (x >= x_min and x < x_max) and (y >= y_min and y < y_max):   
            allowed_len = length                                            
        else:                                                               
            break                                                           
        length += 1                                                         
                                                                             
    x =  int(center[0] + allowed_len * math.cos(angle * np.pi / 180.0)) 
    y =  int(center[1] + allowed_len * math.sin(angle * np.pi / 180.0)) 
    img = img.astype(np.uint8)                                              
    img = cv2.line(img, (center[0], center[1]), (x,y), (255,255, 255), 1)
    return img

# Getting the mires given the img and the line at an angle
def process_mires(img, center, angle, weights=None):
    
    mask = plot_line(img.copy(), center, angle)
    line = mask.copy()
    mask[mask > 0] = 1
    masked_img = img * mask
    connectivity = 4  # You need to choose 4 or 8 for connectivity type
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(masked_img , connectivity , cv2.CV_32S)
    centroids = [list(x) for x in list(centroids)]
    if weights is not None:
        centroids = []
        for idx in np.unique(labels):
            y,x = np.argwhere(labels==idx)[:,0], np.argwhere(labels==idx)[:,1]
            centroids.append([np.average(x, weights=weights[y,x]), np.average(y, weights=weights[y,x])])

    centroids_new = []
    for centroid in centroids:
        r = math.sqrt((center[0]-centroid[0])**2 + (center[1]-centroid[1])**2)
        centroid.append(r)
        centroids_new.append(centroid)
    return list(centroids_new[1:]), line

def plot_highres(image_cent_list, center, n_mires, jump, start_angle, end_angle):
    plt.figure()
    for mire in range(n_mires):
        x_axis, y_axis = [], []
        for idx, angle in enumerate(np.arange(start_angle, end_angle, jump)):
            if len(image_cent_list[idx]) <= mire:
                continue
            else:
                x_axis.append(image_cent_list[idx][mire][0])
                y_axis.append(image_cent_list[idx][mire][1])
        plt.scatter(x_axis, y_axis, s=1, marker='.', linewidths=0, label=str(mire))
    plt.legend()
    #plt.savefig('out/plots/'+'mires.svg', format='svg', dpi=1600)
    plt.savefig('out/plots/'+'weighted-mires.png', format='png', dpi=1200)
    plt.close()

def increase_res(image_gray, image_seg, image_edge, center, ups, image_name):
    image_gray = cv2.resize(image_gray, (image_gray.shape[1]*ups, image_gray.shape[0]*ups), interpolation=cv2.INTER_LINEAR)
    image_seg = cv2.resize(image_seg, (image_seg.shape[1]*ups, image_seg.shape[0]*ups), interpolation=cv2.INTER_NEAREST)
    image_edge = cv2.resize(image_edge, (image_edge.shape[1]*ups, image_edge.shape[0]*ups), interpolation=cv2.INTER_NEAREST)
    center = (center[0]*ups, center[1]*ups)
    cv2.imwrite('out/'+image_name+'_out_gray.png', image_gray)
    cv2.imwrite('out/'+image_name+'_out_clean_inv.png', image_seg)
    cv2.imwrite('out/'+image_name+'_out_edge_inv_filter.png', image_edge)
    return image_gray, image_seg, image_edge, center

def plot_points(X,Y,Z, title='test', zlim=[-1,2]):

    # filter nan values
    check = np.isnan(Z)
    Z[check] = 0
    
    fig = plt.figure(figsize=(12, 8), dpi=80)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=__cm__.RdYlGn,\
                                linewidth=0, antialiased=False, alpha = 0.6)
    v = max(abs(Z.max()),abs(Z.min()))
    ax.set_zlim(zlim[0], zlim[1])
    ax.zaxis.set_major_locator(__LinearLocator__(10))
    ax.zaxis.set_major_formatter(__FormatStrFormatter__('%.02f'))
    cset = ax.contourf(X, Y, Z, zdir='z', offset=-1, cmap=__cm__.RdYlGn)
    fig.colorbar(surf, shrink=1, aspect=30)
    plt.title(title,fontsize=16)
    plt.show()

    # plot isocontour map
    contours = plt.contour(X, Y, Z, 3, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)
    plt.imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap='RdGy', alpha=0.5)
    plt.colorbar()
    plt.show()

def get_three_d_points(y, z, angle, origin=(0,0)):
    if angle < 90:
        x = np.abs(y*math.cos(angle*np.pi/180.0))
        y = np.abs(y*math.sin(angle*np.pi/180.0))  
    elif angle >= 90 and angle < 180:
        angle = 180-angle
        x = -np.abs(y*math.cos(angle*np.pi/180.0))
        y = np.abs(y*math.sin(angle*np.pi/180.0))
    elif angle >= 180 and angle < 270:
        angle = angle-180
        x = -np.abs(y*math.cos(angle*np.pi/180.0))
        y = -np.abs(y*math.sin(angle*np.pi/180.0))
    else:
        angle = 360-angle
        x = np.abs(y*math.cos(angle*np.pi/180.0))
        y = -np.abs(y*math.sin(angle*np.pi/180.0))

    #magnitude = math.sqrt(x**2 + y**2 + z**2)
    magnitude = 1.0

    return x/magnitude, y/magnitude, z/magnitude

def draw_circles(img, center, radii, angle, sim_k):
    img = cv2.line(img, (center[0]-10, center[1]), (center[0]+10, center[1]), (255,255, 255), 2)
    img = cv2.line(img, (center[0], center[1]-10), (center[0], center[1]+10), (255,255, 255), 2)
    for r in radii:
        img = cv2.circle(img, center, r, (5, 5, 5), 1)

    # drawing the angle
    x1, y1 =  int(center[0] + radii[0] * math.cos(angle)) , int(center[1] + radii[0] * math.sin(angle)) 
    x2, y2 =  int(center[0] - radii[0] * math.cos(angle)) , int(center[1] - radii[0] * math.sin(angle))
    img = cv2.line(img, (x1, y1), (x2, y2), (5, 5, 5), 1)

    # writing the sim_k1 value
    x2, y2 =  int(center[0] - 30 * math.cos(angle)) , int(center[1] - 30 * math.sin(angle))
    cv2.putText(img, str(sim_k[0]), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255),1)

    # drawing perpendicular
    angle = angle + np.pi/2
    x1, y1 =  int(center[0] + radii[0] * math.cos(angle)) , int(center[1] + radii[0] * math.sin(angle)) 
    x2, y2 =  int(center[0] - radii[0] * math.cos(angle)) , int(center[1] - radii[0] * math.sin(angle))
    img = cv2.line(img, (x1, y1), (x2, y2), (5, 5, 5), 1)

    # writing sim_k2 value
    x1, y1 =  int(center[0] + 30 * math.cos(angle)) , int(center[1] + 30 * math.sin(angle))
    x2, y2 =  int(center[0] - 30 * math.cos(angle)) , int(center[1] - 30 * math.sin(angle))
    if x1 < x2:
        x2 , y2 = x1, y1
    cv2.putText(img, str(sim_k[1]), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255),1)

    return img