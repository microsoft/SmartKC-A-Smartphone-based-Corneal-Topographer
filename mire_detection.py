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
    n_mires=20, jump=2, start_angle=0, end_angle=360, output_folder="out",
    heuristics_cleanup_flag=True,
    heuristics_bump_cleanup_flag = True):
    
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
    
    if (heuristics_bump_cleanup_flag):
        r_pixels = bump_cleanup_heuristics(r_pixels, output_folder, image_name)
    
    if (heuristics_cleanup_flag): 
        r_pixels = cleanup_plots_heuristics(r_pixels, start_angle, end_angle, jump, n_mires, output_folder, image_name)
    
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

def cleanup_plots_heuristics(r_pixels, start_angle, end_angle, jump, n_mires, output_folder, image_name):

    # Trying to ensure that the plots.png has smooth values: 
    # (a) no value of zero, 
    # (b) no higher mire touching a lower mire, 
    # (c) no random peaks in one mire,

    plt.figure()
    r_pixels_cleaned = []
    for idx, angle in enumerate(np.arange(start_angle, end_angle, jump)):
        # Reshaping from mires x angle to angle x mires
        r_pixels_angle = []
        for mire in range(n_mires):
            r_pixels_angle.append(r_pixels[mire][idx])
        diff_val = [item - r_pixels_angle[idx1 - 1] for idx1, item in enumerate(r_pixels_angle)][1:]

        # Fixing (a) no value of zero
        for i, val in enumerate(r_pixels_angle):
            if val==0:
                diff_val = [item - r_pixels_angle[idx1 - 1] for idx1, item in enumerate(r_pixels_angle[:i])][1:]
                r_pixels_angle[i] = r_pixels_angle[i-1] + np.mean(diff_val)

        # Fixing (b) no higher mire touching a lower mire
        flag = True
        while flag:
            for i, val in enumerate(diff_val):
                if val < 1:
                    if i==1:
                        diff_val[i] = diff_val[0]
                    else:
                        diff_val[i] = np.mean(diff_val[:i-1])
                    r_pixels_angle[i+1] = r_pixels_angle[i+1] + diff_val[i]
                    diff_val = [item - r_pixels_angle[idx1 - 1] for idx1, item in enumerate(r_pixels_angle)][1:]            
            flag = False
            for i, val in enumerate(diff_val):
                if val < 1:
                    flag = True
        
        # Fixing (c) no random peaks in one mire      
        diff_val = [item - r_pixels_angle[idx1 - 1] for idx1, item in enumerate(r_pixels_angle)][1:]
        while np.std(diff_val)>2:
            max_idx = diff_val.index(max(diff_val))
            r_pixels_angle[max_idx+1] = r_pixels_angle[max_idx] + np.mean(diff_val)
            diff_val = [item - r_pixels_angle[idx1 - 1] for idx1, item in enumerate(r_pixels_angle)][1:]
        r_pixels_cleaned.append(r_pixels_angle)

    # Reshaping from angle x mires to mires x angle 
    r_pixels = []
    for mire in range(n_mires):
        r_pixels_temp = []
        for idx, angle in enumerate(np.arange(start_angle, end_angle, jump)):
            r_pixels_temp.append(r_pixels_cleaned[idx][mire])
        plt.plot(np.arange(start_angle, end_angle, jump), r_pixels_temp, ls='-', label=str(mire))
        r_pixels.append(r_pixels_temp)
    plt.legend()
    plt.savefig(output_folder+'/'+image_name+'/plots-modified.png')
    plt.close()
    
    return r_pixels


# Nipun's code

def bump_cleanup_heuristics(r_pixels, output_folder, image_name, threshold=2):
    smoothed_seqs = []
    for i, v in enumerate(r_pixels):
    #   plot_filename = f"plots_bump/plot_{i}"
    #   print(f"plot_bump_filename: {plot_filename}")
      smoothed_v = detect_and_replace_bumps(i, v, threshold, r_pixels)
      smoothed_seqs.append(smoothed_v)
    plot_array_list(smoothed_seqs, [f"{i}" for i in range(len(smoothed_seqs))], output_folder+'/'+ image_name +"/" + 'bump_smoothened.png')
    return smoothed_seqs

def detect_bumps(array, threshold=2):
  # compute the derivative of the array
  derivative = np.gradient(array)
  # define constants for the point types
  HIGH_POSITIVE = "high positive"
  HIGH_NEGATIVE = "high negative"
  SOFT = "soft"
  # initialize an empty list to store the bump regions
  bump_regions = []
  # initialize a variable to store the start index of the current bump region
  start_index = None
  # initialize a variable to store the type of the current bump region (positive or negative)
  bump_type = None
  # loop through the derivative array
#   print(array.shape, "INSIDE detect_bumps: array SHAPE")
  for i in range(len(derivative)):
    # get the current value
    value = derivative[i]
    # check if it is a high positive derivative point
    if value > threshold:
      # mark it as such
      point_type = HIGH_POSITIVE
    # check if it is a high negative derivative point
    elif value < -threshold:
      # mark it as such
      point_type = HIGH_NEGATIVE
    # otherwise, it is a soft-derivative point
    else:
      # mark it as such
      point_type = SOFT
    # check if we are at the first element
    if i == 0:
      # store the current point type as the previous one
      prev_point_type = point_type
    # otherwise, compare with the previous point type
    else:
      # if a soft point is followed by a high positive derivative point
      if prev_point_type == SOFT and point_type == HIGH_POSITIVE and start_index is None:
        # set the start index to the current index
        start_index = i
        # set the bump type to positive
        bump_type = "positive"
      # if a soft point is followed by a high negative derivative point
      elif prev_point_type == SOFT and point_type == HIGH_NEGATIVE and start_index is None:
        # set the start index to the current index
        start_index = i
        # set the bump type to negative
        bump_type = "negative"
      # if a high negative derivative point is followed by a soft point and start index is not None and bump type is positive
      elif prev_point_type == HIGH_NEGATIVE and point_type == SOFT and start_index is not None and bump_type == "positive":
        # append a pair of start and end indices and bump type to the list 
        bump_regions.append((start_index, i, bump_type))
        # reset the start index to None and bump type to None
        start_index = None
        bump_type = None
      # if a high positive derivative point is followed by a soft point and start index is not None and bump type is negative 
      elif prev_point_type == HIGH_POSITIVE and point_type == SOFT and start_index is not None and bump_type == "negative":
        # append a pair of start and end indices and bump type to the list 
        bump_regions.append((start_index, i, bump_type))
        # reset the start index to None and bump type to None
        start_index = None
        bump_type = None      
      # update the previous point type with the current one
      prev_point_type = point_type

  # check if there is a pending bump region after the loop and start index is not None and bump type is not None 
  if start_index is not None and bump_type is not None:
    # append a pair of start and end indices and bump type to the list 
    bump_regions.append((start_index, len(derivative) - 1, bump_type))

  return bump_regions, derivative


def replace_bump_data_with_mean_of_rest(idx, data_seq, bump_regions, r_pixels):
  # Compute indices of data_seq points inside the bump regions
  bump_indices = []
  for bump in bump_regions:
      bump_indices.extend(range(bump[0], bump[1] + 1))
  
  # Compute indices of data_seq points outside the bump regions
  outside_bump_indices = [i for i in range(len(data_seq)) if i not in bump_indices]
  
  # Compute mean of data_seq points outside the bump regions
#   temp = [data_seq]
  mean = np.mean([data_seq[i] for i in outside_bump_indices])
#   mean = np.mean(data_seq[outside_bump_indices])

  # Replace data_seq points inside the bump regions with the mean
#   outside_bump_indices.sort()
  if(idx > 0):
    diff = []
    for i in outside_bump_indices:
        val = r_pixels[idx-1][i] - r_pixels[idx-2][i]
        if val < 0: val = r_pixels[0][i]
        diff.append(val)
    median = np.median(val)
    print(median, "median")
    for i in bump_indices:
      for bump in bump_regions:
        if bump[0] <=i <= bump[1]:
          if bump[1]+1 < len(data_seq):
            mean = (data_seq[bump[0]-1] + data_seq[bump[1]+1])/2
          else:
            mean = data_seq[bump[0]-1]
            break
            # median = np.median([max(0, r_pixels[idx][j] - r_pixels[idx-1][j]) for j in range(0, i)])
            # mean = np.mean([r_pixels[idx][j] for j in range(i-50, min(i+50, len(data_seq)))])
            # data_seq[i] = mean
      data_seq[i] = mean
#   data_seq[bump_indices] = mean
  return data_seq

def plot_two_arrays(array1, array2, legend1, legend2, output_file):
  plt.plot(array1, label=legend1)
  plt.plot(array2, label=legend2)
  plt.legend()
  plt.savefig(output_file)
  plt.close()
  #plt.show()

def plot_array(array1, legend1, output_file):
  print("plotting array")
  plt.plot(array1, label=legend1)
  plt.legend()
  plt.savefig(output_file)
  plt.close()
  #plt.show()
  
def detect_and_replace_bumps(i, data_seq, threshold, r_pixels):
  bump_regions, derivative = detect_bumps(data_seq, threshold)
  print(f"bump_regions: {bump_regions}")

  data_seq_orig = data_seq.copy()
  data_seq = replace_bump_data_with_mean_of_rest(i, data_seq, bump_regions, r_pixels)
#   plot_two_arrays(data_seq_orig, data_seq, "original", "smoothed", f"{plot_filename_prefix}_smoothed.png")
#   plot_array(derivative[2:], "derivative", f"{plot_filename_prefix}_derivative.png")
  return data_seq

def plot_array_list(array_list, legend_list, output_file):
  for i in range(len(array_list)):
    plt.plot(array_list[i], label=legend_list[i])
  plt.legend()
  plt.savefig(output_file)
  plt.close()
  #plt.show()