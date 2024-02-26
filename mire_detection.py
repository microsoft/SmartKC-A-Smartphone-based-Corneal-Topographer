#!/usr/bin/env python
# Author: Siddhartha Gairola (t-sigai at microsoft dot com))
import logging
from constants import Constants
from utils import *
from scipy.signal import medfilt
from scipy.ndimage import median_filter
from constants import Constants

def detect_mires_img_proc(image_seg, image_orig, center, 
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
        cent, line = process_mires(image_seg.copy(), center, angle, weights= None)#weights=image_orig)
        cent_inv, _ = process_mires(image_inv.copy(), center, angle, weights=None)#255-image_orig)
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

def fetch_points_to_mask(r_pixels, coords, flagged_points, n_mires, start_angle, end_angle, jump):
    points_to_mask = []
    mean_radii = []
    for mire in range(n_mires):
        mean_radii.append(np.nanmean(r_pixels[mire]))

    for mire in range(n_mires):
        current_set = []
        for idx, angle in enumerate(np.arange(start_angle, end_angle, jump)):
            if (mire, angle) in flagged_points:
                current_set.append(angle)
            else:
                if len(current_set) > Constants.POSTPROCESS_MASKING_THRESHOLD * jump:
                    points_to_mask.append((min(current_set), max(current_set), mire, mean_radii[mire]))
                current_set = []
    return points_to_mask

def clean_points(image_cent_list, image_gray, image_name, center, mire_loc_method,
    n_mires=20, jump=2, start_angle=0, end_angle=360, output_folder="out",
):
    
    logging.info("Cleaning ...")
    image_gray = np.dstack((image_gray, np.dstack((image_gray, image_gray))))
    coords, r_pixels = [], []
    flagged_points = []
    plt.figure()
    for mire in range(n_mires):
        r_pixels_temp, coords_temp = [], []

        # idx here is traversing the angles
        for idx, angle in enumerate(np.arange(start_angle, end_angle, jump)):
            if len(image_cent_list[idx]) <= mire or np.isnan(image_cent_list[idx][mire][2]):
                r_pixels_temp.append(Constants.UNKNOWN_RADIUS)
                coords_temp.append(Constants.UNKNOWN_COORDINATE)
                if mire_loc_method == Constants.GRAPH_CLUSTER_LOC_METHOD and mire != 0:
                    flagged_points.append((mire-1, angle))
            else:
                x, y = image_cent_list[idx][mire][0], image_cent_list[idx][mire][1]
                r = math.sqrt((center[0]-x)**2 + (center[1]-y)**2)
                r_pixels_temp.append(r)
                coords_temp.append((y,x)) # Note: coords is (y,x) rather than (x,y)

        # remove outliers using median filtering
        r_pixels_temp = median_filter_with_nans(r_pixels_temp, 25, 51)
        # plot radii        
        plt.plot([n for n, r in enumerate(r_pixels_temp) if not np.isnan(r)], [r for r in r_pixels_temp if not np.isnan(r)], "X", markersize=2, label=str(mire))
        # append
        r_pixels.append(r_pixels_temp)
        coords.append(coords_temp)

    plt.savefig(output_folder+'/'+image_name+'/plots.png')
    plt.close()
    
    r_pixels = r_pixels[1:]

    r_pixels, additional_flagged_points = remove_outliers(r_pixels, mire_loc_method, start_angle, end_angle, jump, n_mires-1)
    flagged_points.extend(additional_flagged_points)

    coords_fixed = []
    for mire in range(n_mires-1):
        coords_temp = []
        for idx, angle in enumerate(np.arange(start_angle, end_angle, jump)):
            radius = r_pixels[mire][idx]
            if np.isnan(radius):
                if mire_loc_method == Constants.RADIAL_SCAN_LOC_METHOD:
                    radius = np.nanmean(r_pixels[mire][max(angle-10, 0):min(angle+10, end_angle)])
                else:
                    assert (mire, angle) in flagged_points, f"Flagged point not in flagged_points - mire : {mire}, angle : {angle}"
            if np.isnan(radius):
                if mire_loc_method == Constants.RADIAL_SCAN_LOC_METHOD:
                    radius = np.nanmean(r_pixels[mire])
                else:
                    assert (mire, angle) in flagged_points, f"Flagged point not in flagged_points - mire : {mire}, angle : {angle}"
            x =  center[0] + radius * math.cos(float(angle) * np.pi / 180.0)
            y =  center[1] + radius * math.sin(float(angle) * np.pi / 180.0)
            coords_temp.append((y,x))
        coords_fixed.append(coords_temp)

    # get cleaned mire image
    image_gray = plot_color_rb(image_gray, coords_fixed, flagged_points)
    cv2.imwrite(output_folder+"/" + image_name + "/" + image_name + "_mp_clean.png", image_gray)

    logging.info("Cleaning Done!")

    points_to_mask = fetch_points_to_mask(r_pixels, coords_fixed, flagged_points, n_mires - 1, start_angle, end_angle, jump)
    # note that coords_fixed has order of coords as {(y,x)}
    return r_pixels, flagged_points, coords_fixed, image_gray, points_to_mask

def remove_outliers(r_pixels, mire_loc_method, start_angle, end_angle, jump, n_mires):
    flagged_points = []

    assert len(r_pixels) == n_mires, f"Number of mires is not consistent - r_pixels : {len(r_pixels)}, n_mires : {n_mires}"
    r_pixels_angle = [[] for _ in np.arange(start_angle, end_angle, jump)]

    for _ , radii in enumerate(r_pixels):
        assert len(radii) == (end_angle - start_angle) // jump, f"Number of mire points is not consistent - radii : {len(radii)}, angles : {(end_angle - start_angle) // jump}"
        for idx, radius in enumerate(radii):
            r_pixels_angle[idx].append(radius)
    for idx, angle in enumerate(np.arange(start_angle, end_angle, jump)):
        width =  [r_pixels_angle[idx][i] - r_pixels_angle[idx][i-1] for i in range(1, n_mires)]
        # w is the width between the i and i+1'th mire
        for i,w in enumerate(width):
            if i<5:
                continue
            # i+1'th mire point is more than 3 std away from the i'th mire point
            if not np.isnan(w) and w > np.nanstd(width[:i])*10 + np.nanmean(width[:i]):
                logging.info(f"Found outlier point - {i+1}, {angle}, {w}, {np.nanstd(width[:i])*10 + np.nanmean(width[:i])}")
                # only flagging points if graph_cluster is the mire_loc method, points are not flagged in radial_scan
                if mire_loc_method == Constants.GRAPH_CLUSTER_LOC_METHOD:
                    # ignoring i+1'th mire point
                    flagged_points.append((i+1, angle))
                else:
                    # fixing i+1'th mire point
                    r_pixels_angle[idx][i+1] = r_pixels_angle[idx][i] + np.nanmean(width[:i])

    if mire_loc_method == Constants.GRAPH_CLUSTER_LOC_METHOD:
        r_pixels_corrected = r_pixels
    else:
        r_pixels_corrected = [[] for _ in range(n_mires)]
        for angle in np.arange(start_angle, end_angle, jump):
            for mire in range(n_mires):
                r_pixels_corrected[mire].append(r_pixels_angle[angle][mire])
    
    return r_pixels_corrected, flagged_points
