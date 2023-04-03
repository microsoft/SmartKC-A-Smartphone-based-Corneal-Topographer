#!/usr/bin/env python
# Author: Siddhartha Gairola (t-sigai at microsoft dot com))
import logging
from utils import *
from scipy.signal import medfilt


def medfilt_clean(r_pixels, win_1=25, win_2=51):
    # overfiltering and cleaning of r_pixels using median filtering
    r_pixels_overfilter = np.array(r_pixels).copy()
    r_pixels_overfilter = np.concatenate([r_pixels_overfilter[-win_1:],
        r_pixels_overfilter, r_pixels_overfilter[:win_1]])
    r_pixels_overfilter = r_pixels_overfilter[win_1:win_1+len(r_pixels)]
    r_pixels_overfilter = medfilt(r_pixels_overfilter, win_2)
    r_pixels = medfilt(r_pixels, win_1)

    diff = np.abs(r_pixels-r_pixels_overfilter)/(r_pixels_overfilter+1e-9)
    for idx in range(len(r_pixels)):
        if diff[idx] > 0.05: # cut off is manually set at 5%
            r_pixels[idx] = r_pixels_overfilter[idx]

    return r_pixels

def medfilt_tuple(r_pixels_list, w=11, cutoff_angle=270):
    r_pixels_list.sort(key = lambda x: x[0])
    r_pixels_1 = [r for (angle, r) in r_pixels_list if angle <= cutoff_angle]
    r_pixels_2 = [r for (angle, r) in r_pixels_list if angle > cutoff_angle]
    angles_1 = [angle for (angle, r) in r_pixels_list if angle <= cutoff_angle]
    angles_2 = [angle for (angle, r) in r_pixels_list if angle > cutoff_angle]

    if np.min(angles_1) == 0 and len(angles_2) > 0:
        r_pixels = r_pixels_2 + r_pixels_1
        angles = angles_2 + angles_1
    else:
        r_pixels = r_pixels_1 + r_pixels_2
        angles = angles_1 + angles_2

    r_pixels = medfilt(r_pixels, w)
    out = []
    for idx in range(len(r_pixels)):
        out.append((angles[idx], r_pixels[idx]))

    return out

def curve_fit_fill(r_pixels_list, skip_angles=[], jump=1, deg=2):

    s1, e1 = int(skip_angles[0][0]), int(skip_angles[0][1])
    s2, e2 = int(skip_angles[1][0]), int(skip_angles[1][1])
    r_pixels_list[s1] = 0
    r_pixels_list[s2] = 0
    r_pixels_list[e1] = 0
    r_pixels_list[e2] = 0
    # first gap
    x,y = [],[]
    x_test = []

    left, right = s1-10, (e1+10)%360

    idx = 0
    while True:
        angle = left
        if angle >= 360:
            angle = angle%360
        curr_r = r_pixels_list[angle]
        if curr_r == 0:
            x_test.append(idx)
        else:
            x.append(idx)
            y.append(curr_r)

        if angle == right:
            break

        left += 1
        idx += 1

    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    y_test = p(x_test)
    idx = 0
    left, right = s1, e1
    while True:
        angle = left
        if angle >= 360:
            angle = angle%360
        r_pixels_list[angle] = y_test[idx]
        if angle == right:
            break
        left += 1
        idx += 1

    # second gap
    x, y = [], []
    x_test = []
    left, right = s2-10, e2+10
    idx = 0
    for i in range(left, right+1):
        curr_r = r_pixels_list[i]
        if curr_r == 0:
            x_test.append(idx)
        else:
            x.append(idx)
            y.append(curr_r)
        idx += 1

    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    y_test = p(x_test)
    idx = 0
    for i in range(s2, e2+1):
        r_pixels_list[i] = y_test[idx]
        idx += 1

    return r_pixels_list


def process(image_seg, image_orig, center, 
    jump=2, start_angle=0, end_angle=360):
    logging.info("Processing ...")

    image_or = image_seg.copy() 
    image_and = np.zeros_like(image_seg)
    image_mp = image_orig.copy()
    image_mp = np.dstack((image_mp, np.dstack((image_mp, image_mp))))
    image_cent_list = []

    image_inv = 255 - image_seg
    for angle in np.arange(start_angle, end_angle, jump):
        # edge image processing for angle
        cent, line = process_mires(image_seg.copy(), center, angle, weights=image_orig)
        cent_inv, _ = process_mires(image_inv.copy(), center, angle, weights=255-image_orig)
        cent = cent + cent_inv
        cent.sort(key = lambda x: x[2])
        image_cent_list.append(cent)

        image_or = cv2.bitwise_or(image_or, line, mask=None)
        temp = cv2.bitwise_and(image_seg, line, mask=None)
        image_and = cv2.bitwise_or(image_and, temp, mask=None)
        image_mp = plot_color_rb(image_mp, cent)

    logging.info("Processing Done!")
    # remember image_cent_list is a list of list, that has 
    # centroids with {(x,y)} order and has length as number of angles
    return image_cent_list, center, [image_or, image_and, image_mp]

def clean_points(image_cent_list, image_gray, image_name, center, 
    n_mires=20, jump=2, start_angle=0, end_angle=360, output_folder="out"):
    
    logging.info("Cleaning ...")
    image_gray = np.dstack((image_gray, np.dstack((image_gray, image_gray))))
    coords, r_pixels = [], []
    plt.figure()
    for mire in range(n_mires):
        r_pixels_temp, coords_temp = [], []

        # idx here is traversing the angles
        for idx, angle in enumerate(np.arange(start_angle, end_angle, jump)):

            if len(image_cent_list[idx]) <= mire:
                r_pixels_temp.append(0)
                coords_temp.append((-1,-1))
            else:
                r = math.sqrt((center[0]-image_cent_list[idx][mire][0])**2 
                    + (center[1]-image_cent_list[idx][mire][1])**2)
                x, y = image_cent_list[idx][mire][0], image_cent_list[idx][mire][1]
                r_pixels_temp.append(r)
                coords_temp.append((y,x)) # Note: coords is (y,x) rather than (x,y)

        # remove outliers using median filtering
        r_pixels_temp = medfilt_clean(r_pixels_temp)
        # plot radii
        plt.plot(np.arange(start_angle, end_angle, jump), r_pixels_temp, ls='-', label=str(mire))
        # append
        r_pixels.append(r_pixels_temp)
        coords.append(coords_temp)

    plt.legend()
    plt.savefig(output_folder+'/'+image_name+'/plots.png')
    plt.close()

    # uncomment for real image, skip first mire
    r_pixels = r_pixels[1:]

    coords_fixed = []
    for mire in range(n_mires-1):
        coords_temp = []
        for idx, angle in enumerate(np.arange(start_angle, end_angle, jump)):
            radius = r_pixels[mire][idx]
            x =  center[0] + radius * math.cos(float(angle) * np.pi / 180.0)
            y =  center[1] + radius * math.sin(float(angle) * np.pi / 180.0)
            coords_temp.append((y,x))
        coords_fixed.append(coords_temp)

    # get cleaned mire image
    image_gray = plot_color_rb(image_gray, coords_fixed)
    cv2.imwrite(output_folder+"/" + image_name + "/" + image_name + "_mp_clean.png", image_gray)

    logging.info("Cleaning Done!")
    # note that coords_fixed has order of coords as {(y,x)}
    return r_pixels, coords_fixed, image_gray

def clean_points_support(image_cent_list, image_gray, image_name, 
    center, n_mires=20, jump=2, start_angle=0, end_angle=360, 
    skip_angles=[], output_folder="out"):

    # NOTE: THIS IS HAS NOT BEEN TESTED WELL ENOUGH
    logging.info("Cleaning ...")
    image_gray = np.dstack((image_gray, np.dstack((image_gray, image_gray))))

    r_pixels = []
    plt.figure()
    for mire in range(n_mires):
        r_pixels_1, r_pixels_2, r_pixels_3, r_pixels_4 = [], [], [], []
        r_pixels_temp = -1

        # idx here is traversing the angles
        for idx, angle in enumerate(np.arange(start_angle, end_angle, jump)):
            if len(image_cent_list[idx]) <= mire:
                r_pixels_temp = 0
            else:
                r = math.sqrt((center[0]-image_cent_list[idx][mire][0])**2 
                    + (center[1]-image_cent_list[idx][mire][1])**2)
                x, y = image_cent_list[idx][mire][0], image_cent_list[idx][mire][1]
                r_pixels_temp = r

            zone = check_angle(angle, skip_angles)
            if zone == 1:
                r_pixels_1.append((angle, 0))
            elif zone == 2:
                r_pixels_2.append((angle,r_pixels_temp))
            elif zone == 3:
                r_pixels_3.append((angle,0))
            elif zone == 4:
                r_pixels_4.append((angle, r_pixels_temp))

        # remove outliers using median filtering
        r_pixels_2 = medfilt_tuple(r_pixels_2, w=25)
        r_pixels_4 = medfilt_tuple(r_pixels_4, w=25)

        r_pixels_temp = r_pixels_1 + r_pixels_2 + r_pixels_3 + r_pixels_4
        r_pixels_temp.sort(key = lambda x: x[0])
        r_pixels_temp = [r for (angle, r) in r_pixels_temp]

        r_pixels_temp = curve_fit_fill(r_pixels_temp, skip_angles=skip_angles)

        plt.plot(np.arange(start_angle, end_angle, jump), r_pixels_temp, ls='-', label=str(mire))

        # append
        r_pixels.append(r_pixels_temp)

    plt.legend()
    #plt.savefig('out/'+image_name+'/plots.png')
    plt.close()

    # skip first mire
    # uncomment for real image
    r_pixels = r_pixels[1:]

    coords_fixed = []
    for mire in range(n_mires-1):
        coords_temp = []
        for idx, angle in enumerate(np.arange(start_angle, end_angle, jump)):
            radius = r_pixels[mire][idx]
            x =  center[0] + radius * math.cos(float(angle) * np.pi / 180.0)
            y =  center[1] + radius * math.sin(float(angle) * np.pi / 180.0)
            coords_temp.append((y,x))
            zone = check_angle(angle, skip_angles)
            if zone == 1 or zone == 3:
                image_gray[int(y), int(x), :] = [0, 255, 0]
            elif mire%2 == 0:
                image_gray[int(y), int(x), :] = [0, 0, 255]
            else:
                image_gray[int(y), int(x), :] = [0, 255, 255] # yellow

        coords_fixed.append(coords_temp)

    #cv2.imwrite(output_folder+'/'+image_name+'/'+image_name+'_mp_fitted.png', image_gray)

    logging.info("Cleaning Done!")
    return r_pixels, coords_fixed, image_gray