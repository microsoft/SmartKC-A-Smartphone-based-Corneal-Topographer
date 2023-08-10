import numpy as np
import cv2
from constants import Constants
from get_center.get_center import colors_list
from get_center.get_center import segment_and_get_center
import os

from utils import generate_colored_image

def detect_mires_mask_dl(image_gray, mask_output_dir):
    image_gray_copy = image_gray.copy()
    image_gray_with_channels = np.dstack((image_gray.copy(), np.dstack((image_gray_copy, image_gray_copy))))
    # generate mask
    mire_mask = get_mask(image_gray_with_channels)
    mire_mask = mire_mask.astype(int)
    #dump mask
    generate_colored_image(mire_mask, mask_output_dir + "/" + "mire_mask.png")
    np.savetxt(mask_output_dir + "/" + "mire_mask.csv", mire_mask, delimiter=",", fmt="%d")
    #return mask and image with channels
    return mire_mask, image_gray_with_channels

    # 'mask' is a WxH image, where each pixel has either a value of 0 if it is a
    # background pixel, or a value of i if it is on the i'th mire.
    # 'center' is a tuple (x, y) of the center of the mire pattern.

    # This function returns an 2D array -- radii, where radii[mire_index][angle]
    # gives the radius of the 'mire_index' mire at angle 'angle'.
    # To compute the radius, we first compute the intersection of the radial
    # line at angle 'angle' from the 'center' with the 'mire_index' mire. The
    # radius is then the distance from the center to the mid-point of this
    # intersection.

    radii = np.full((n_mires, num_angles), -1)

    # Compute the angle of each pixel from the center.
    x = np.arange(mask.shape[1]) - center[0]
    y = np.arange(mask.shape[0]) - center[1]
    xx, yy = np.meshgrid(x, y)
    angles = np.arctan2(yy, xx)
    angles[angles < 0] += 2 * np.pi

    # Compute the distance of each pixel from the center.
    distances = np.sqrt(xx**2 + yy**2)

    # Compute the radius of each mire at each angle.
    for mire_index in range(n_mires):
        # Find the pixels that are on the mire.
        mire_pixels = np.where(mask == mire_index + 1)

        # Find the angle of each pixel from the center.
        mire_angles = angles[mire_pixels]

        # Find the distance of each pixel from the center.
        mire_distances = distances[mire_pixels]

        # Find the mid-point of the intersection of the radial line at each
        # angle with the mire.
        mire_radii = np.zeros(num_angles)
        for angle_index in range(num_angles):
            angle = angle_index * 2 * np.pi / num_angles
            #angle_distances = mire_distances[np.where(np.isclose(mire_angles, angle))]
            tolerance = 2*np.pi/360.
            angle_distances = mire_distances[np.where(np.isclose(mire_angles, angle, atol = tolerance))]
            #print(f"angle: {angle}, len(angle_distances): {len(angle_distances)}")
            if len(angle_distances) > 0:
                mire_radii[angle_index] = np.median(angle_distances)

        # Store the radii.
        radii[mire_index] = mire_radii

    return radii


def segmentation_mask_to_radii_ray(mask, center, n_mires, num_angles):
    # 'mask' is a WxH image, where each pixel has either a value of 0 if it is a
    # background pixel, or a value of i if it is on the i'th mire.
    # 'center' is a tuple (x, y) of the center of the mire pattern.

    # This function returns an 2D array -- radii, where radii[mire_index][angle]
    # gives the radius of the 'mire_index' mire at angle 'angle'.
    # To compute the radius, we first compute the intersection of the radial
    # line at angle 'angle' from the 'center' with the 'mire_index' mire. The
    # radius is then the distance from the center to the mid-point of this
    # intersection.

    radii = np.full((n_mires, num_angles), Constants.UNKNOWN_RADIUS)

    # Compute the distance of each pixel from the center.
    x = np.arange(mask.shape[1]) - center[0]
    y = np.arange(mask.shape[0]) - center[1]
    xx, yy = np.meshgrid(x, y)

    distances = np.sqrt(xx**2 + yy**2)

    # Compute the radius of each mire at each angle.
    count_found = 0
    count_notfound = 0
    for angle_index in range(num_angles):
        angle = angle_index * 2 * np.pi / num_angles
        x_step = np.cos(angle)
        y_step = np.sin(angle)

        current_mire_index = 0
        tracking_mask_value = 1
        tracking_zero = False
        # for mire_index in range(n_mires):
        # Initialize a numpy array to save the pixel indices of pixels which lie on the current mire and angle

        # Find the mid-point of the intersection of the radial line at each
        # angle with the mire.
        x = center[0]
        y = center[1]
        pixel_indices = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)
        while True:
            # if n_mires reached, break
            if(current_mire_index > n_mires-1): break
            x = x + x_step
            y = y + y_step
            x_int = int(round(x))
            y_int = int(round(y))

            if x_int < 0 or x_int >= mask.shape[1] or y_int < 0 or y_int >= mask.shape[0]:
                # we have reached the end of the ray
                break

            if not tracking_zero and mask[y_int, x_int] == tracking_mask_value:
                # This is a pixel on the currently tracking (non-zero) mire.
                # add this pixel to pixel_indices
                pixel_indices[y_int, x_int] = True
                #print(f"Found pixel for non zero mire: tracking_mire_index: {tracking_mask_value}, angle_index: {angle_index}, angle: {angle}")
            elif tracking_zero and mask[y_int, x_int] == 0:
                # This is a pixel on the currently tracking (zero) mire.
                pixel_indices[y_int, x_int] = True
                #print(f"Found pixel for zero mire: tracking_mire_index: {tracking_mask_value}, angle_index: {angle_index}, angle: {angle}")
            elif not tracking_zero and mask[y_int, x_int] == 0:
                # we have reached the end of the current (non-zero) mire
                # Commit the current mire to radii

                # Take average of mire_distances of all pixels on the mire and angle
                if len(distances[pixel_indices]) == 0:
                    #print(f"Found no pixels for non zero mire: tracking_mire_index: {tracking_mask_value}, angle_index: {angle_index}, angle: {angle}")
                    count_notfound += 1
                else:
                    #print(f"mire_index: {mire_index}, angle_index: {angle_index}, angle: {angle}, len(distances[pixel_indices]): {len(distances[pixel_indices])}")
                    mire_radii = np.mean(distances[pixel_indices])
                    radii[current_mire_index][angle_index] = mire_radii
                    count_found += 1

                pixel_indices = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)
                # Increment counters
                tracking_mask_value += 1
                current_mire_index += 1
                tracking_zero = True


            elif tracking_zero and mask[y_int, x_int] == tracking_mask_value:
                # we have reached the end of the current (zero) mire
                # Commit the current mire to radii

                # Take average of mire_distances of all pixels on the mire and angle
                if len(distances[pixel_indices]) == 0:
                    # print(f"Found no pixels for zero mire: tracking_mire_index: {tracking_mask_value}, angle_index: {angle_index}, angle: {angle}")
                    count_notfound += 1
                else:
                    #print(f"mire_index: {mire_index}, angle_index: {angle_index}, angle: {angle}, len(distances[pixel_indices]): {len(distances[pixel_indices])}")
                    mire_radii = np.mean(distances[pixel_indices])
                    radii[current_mire_index][angle_index] = mire_radii
                    count_found += 1

                pixel_indices = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)
                # Increment counters
                current_mire_index += 1
                tracking_zero = False

            else:
                # TODO: Ideally we should continue with the next mire index we see.
                # print(f"Error: Found unexpected mask value: angle: {angle_index}, tracking_mask_value: {tracking_mask_value}, tracking_zero: {tracking_zero}, {mask[y_int, x_int]} at x: {x}, y: {y}, x_int: {x_int}, y_int: {y_int}")
                break


    # print(f"count_found: {count_found}, count_notfound: {count_notfound}")
    return radii


def plot_mire_points_on_image(img, radii, center, n_mires, num_angles):
    # Plot the mire points on the image with yellow dots.
    for mire_index in range(n_mires):
        for angle_index in range(num_angles):
            angle = angle_index * 2 * np.pi / num_angles
            radius = radii[mire_index][angle_index]
            if np.isnan(radius):
                continue
            print(radius, angle, np.isnan(radius), 'radius, angle')
            x = int(center[0] + radius * np.cos(angle))
            y = int(center[1] + radius * np.sin(angle))
            #print(f"mire_index: {mire_index}, angle_index: {angle_index}, angle: {angle}, radius: {radius}, x: {x}, y: {y}, img[y, x]: {img[y, x]}")
            img[y, x] = colors_list[(mire_index)%15]

    return img

def getCentList(radii, n_mires, center, start_angle = 0, end_angle=360, jump = 1):
    coords = []
    image_cent_list = [[-1 for i in range(n_mires)] for j in range(360)]
    
    for idx, angle in enumerate(np.arange(start_angle, end_angle, jump)):
        for mire_index in range(n_mires):
            radius = radii[mire_index][idx]
            image_cent_list[idx][mire_index] = (int(center[0] + radius * np.cos(angle)), int(center[1] + radius * np.sin(angle)))
    return image_cent_list
            
    

def detect_mires_from_mask(mask, center, n_mires, src_image = None, num_angles = 360):
    center = (int(center[0]), int(center[1]))
    y, x = np.argwhere(mask == 1)[:,0], np.argwhere(mask == 1)[:,1]
    y, x = np.mean(y), np.mean(x)

    radii = segmentation_mask_to_radii_ray(mask, center, n_mires, num_angles)
    
    # Only plot if src image path is provided
    overlay_image = plot_mire_points_on_image(src_image, radii, center, n_mires, num_angles)
    
    # image_cent_list = getCentList(radii, n_mires, center)
    return radii, overlay_image

def get_mask(cropped_img):
    script_dir = os.path.dirname(__file__)
    get_center_obj = segment_and_get_center(script_dir+'/get_center/segment_and_get_center_epoch_557_iter_14.pkl')
    mask, _ = get_center_obj.segment(cropped_img)
    mask = mask.astype(int)
    return mask