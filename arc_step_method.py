import sys
sys.path.append("../../")

import numpy as np
from constants import Constants
from hardware.hardware_constants import HardwareConstants
import math
from utils import check_angle, get_three_d_points, draw_circles
import matplotlib.pyplot as plt
from zernike_smartkc import RZern
from get_maps import generate_tan_map, gt_pal, gt_r, generate_axial_map, gt_p
from metrics import compute_simk, compute_tilt_factor, clmi_ppk
import logging

class arc_step_method:

    def __init__(self, placido_file, start_angle, end_angle, jump, skip_angles) -> None:
        self.placido_file = placido_file
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.jump = jump
        self.skip_angles = skip_angles

        self.ring_location_midpoints = self.read_placido_file(placido_file, mid_point=True)
        self.ring_locations = self.read_placido_file(placido_file)

    
    def convert_to_dict(self, r_pixels, coords, flagged_points):
        radii_angle_mires = dict()
        points_angle_mires = dict()

        for mire_num, coord_set in enumerate(coords):
            for angle, c in enumerate(coord_set):
                if (mire_num, angle) in flagged_points:
                    continue
                if angle not in radii_angle_mires:
                    radii_angle_mires[angle] = dict()
                if angle not in points_angle_mires:
                    points_angle_mires[angle] = dict()
                radii_angle_mires[angle][mire_num] = r_pixels[mire_num][angle]
                points_angle_mires[angle][mire_num] = c     
        return points_angle_mires, radii_angle_mires   
    
    def read_placido_file(self, filename, mid_point=False):
        file = open(filename, 'r')
        ring_radius_height = []
        for idx, line in enumerate(file):
            # skipping the first mire since it is black, we are also removing the centermost portion of the image
            if idx == 0:
                continue
            line = line.strip().split()
            ring_radius_height.append([float(line[0]), float(line[1])])
        file.close()

        # calculating mid-points of rings
        if mid_point:
            ring_locations = []
            for idx in range(len(ring_radius_height)-1):
                ring_locations.append((
                (ring_radius_height[idx][0] + ring_radius_height[idx + 1][0] - HardwareConstants.placido_thickness_mm + HardwareConstants.thickness_adjustment_mm) / 2.0, 
                (ring_radius_height[idx][1] + ring_radius_height[idx + 1][1]) / 2.0,
                ))
            return ring_locations                    
        else:
            return ring_radius_height
    
    def fetch_object_slopes(self, size_pixels, image_size, sensor_dims=(6.4, 4.8), f_len=4.76):
        """
        Given the size of mires in terms of pixels and the sensor dimensions, this function finds the ratio of the size of the image formed vs the distance from the smartphone camera lens
        1. The image within the smartphone camera is formed on the sensor, which is at a distance of 1 x f_len from the lens
        2. The sensor dims provide the size of each pixel in the horizontal and vertical directions

        The function - 
        1. Finds the size of the image formed using the sensor dimensions
        2. Finds the ratio of the size of the image formed vs the distance from the smartphone camera lens
        """
        object_dims = (
            size_pixels[0] * sensor_dims[0] / image_size[0],
            size_pixels[1] * sensor_dims[1] / image_size[1],            
        )
        object_size = (object_dims[0] ** 2 + object_dims[1] ** 2) ** 0.5

        slope = object_size / f_len
        return slope


    def fetch_arc_step_params(self, size_pixels, width, height, sensor_dims, f_len, working_distance, mid_point=False):
        if mid_point:
            ring_locations = self.ring_location_midpoints
        else:
            ring_locations = self.ring_locations
        
        k, oz, oy = dict(), dict(), dict()
        for idx, (ring_radius, ring_height) in enumerate(ring_locations):
            if idx not in size_pixels:
                continue

            slope = self.fetch_object_slopes(size_pixels[idx], (width, height), sensor_dims, f_len)
            k[idx+1] = slope
            oy[idx+1] = ring_radius
            oz[idx+1] = ring_height - working_distance
    
        # k, oz, oy = [k[0]] + k, [oz[0]] + oz, [oy[0]] + oy
        return k, oz, oy
    
    def construct_surface(self, nrings, p, oz, oy, k):

        rocs, zs, ys = dict(), dict(), dict()
        for i in range(0, nrings+1):
            if i not in k:
                continue
            step = 0.01
            if i < 2:
                z_old = 0
                y_old = 0
                slope_old = 0
                z = 0
                # z = 0
            if i > 0:
                z = z_old + slope_old*(y-y_old) + 0.5*quad_old*((y-y_old)**2)

            checker_counter = 0
            while True:
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
            zs[i] = z
            ys[i] = y

            curv = abs(quad_old) / (1 + slope**2)**1.5 # curvature
            roc = 1/curv # radius of curvature
            rocs[i] = roc
        return rocs, zs, ys

    
    def run(self, image_seg, image_name, center, r_pixels, coords, height, width, working_distance, sensor_dims=(6.4, 4.8), f_len=4.76,  gap_top = 0, gap_base = 0, flagged_points = [], n_mires = 22):
        """
        mire_locations is a dict - first indexed by angle. mire_locations[angle] is a dict indexed mire number. mire_locations[angle][mire_number] is a tuple of (y, x) coordinates
        """
        logging.info("Running arc step method")
        mire_locations, _ = self.convert_to_dict(r_pixels, coords, flagged_points)


        relative_points = dict()
        # slope values for all angles - shape = 360, x, x is the number of mires seen in the current angle
        arc_step_k = dict()
        ozs, oys = dict(), dict()
        corneal_surface = dict()

        plot_x, plot_y, plot_z = [],[],[]

        rocs_map = np.full((image_seg.shape[0], image_seg.shape[1]), -1e6, dtype="float64")
        elevation_map = rocs_map.copy()

        max_radius = -1

        for angle in np.arange(self.start_angle, self.end_angle, self.jump):

            size_pixels = dict()
            if angle not in mire_locations:
                continue
            relative_points[angle] = dict()
            for mire_number in range(n_mires):
                if mire_number not in mire_locations[angle]:
                    assert (mire_number, angle) in flagged_points, f"Flagged points calculated incorrectly, {(mire_number, angle)} missing in flagged points"
                    continue
                y,x = mire_locations[angle][mire_number]
                radius = np.sqrt((x-center[0])**2 + (y-center[1])**2)
                max_radius = max(max_radius, radius)
                x_diff, y_diff = x - center[0], y - center[1]
                size_pixels[mire_number] = [abs(x_diff), abs(y_diff)]
                relative_points[angle][mire_number] = (y_diff, x_diff)
            
            k , oz, oy = self.fetch_arc_step_params(size_pixels, width, height, sensor_dims, f_len, working_distance + gap_top, mid_point=True)

            arc_step_k[int(angle)] = k
            ozs[int(angle)] = oz
            oys[int(angle)] = oy
        

        # with open(f"../{image_name}_arc_step_k.json", "w") as f:
        #     s = json.loads(json.dumps(arc_step_k))
        #     json.dump(s, f)
        # with open(f"../{image_name}_ozs.json", "w") as f:
        #     s = json.loads(json.dumps(ozs))
        #     json.dump(s, f)
        # with open(f"../{image_name}_oys.json", "w") as f:
        #     s= json.loads(json.dumps(oys))
        #     json.dump(s, f)
        
        for _ , angle in enumerate(np.arange(self.start_angle, self.end_angle, self.jump)):
            if angle not in arc_step_k or 1 not in arc_step_k[angle]:
                continue
            corneal_surface[angle] = dict()            
            opposite_angle = (angle + 180) % 360
            if opposite_angle in arc_step_k:
                if 1 in arc_step_k[angle] and 1 in arc_step_k[opposite_angle]:
                    arc_step_k[angle][0] = (arc_step_k[angle][1] + arc_step_k[opposite_angle][1])/2.0
                elif 1 not in arc_step_k[angle] and 1 in arc_step_k[opposite_angle]:
                    arc_step_k[angle][0] = arc_step_k[opposite_angle][1]
                elif 1 in arc_step_k[angle] and 1 not in arc_step_k[opposite_angle]:
                    arc_step_k[angle][0] = arc_step_k[angle][1]                    
                else:
                    logging.info(f"Skipping angle - {angle}, {opposite_angle}")
                    continue

            elif 1 not in arc_step_k[angle]:                
                logging.info(f"Skipping angle - {angle}")
                continue

            k = arc_step_k[angle]
            oys[angle][0] = oys[angle][1]
            ozs[angle][0] = ozs[angle][1]
            zone = check_angle(angle, self.skip_angles)
            if zone == 1 or zone == 3:
                continue

            try:
                rocs, zs, ys = self.construct_surface(
                    n_mires, -(working_distance + gap_top + gap_base), ozs[angle], oys[angle], arc_step_k[angle]
                )
            except:
                continue

            for mire_num, mire_location in mire_locations[angle].items():
                if mire_num not in rocs:
                    continue
                y_float, x_float = mire_location
                y = int(y_float)
                x = int(x_float)
                rocs_map[y, x] = rocs[mire_num+1]
                elevation_map[y, x] = zs[mire_num+1]

                x_, y_, z = get_three_d_points(ys[mire_num+1], zs[mire_num+1], angle)
                # assert abs(x_float - x_ ) <= 0.001, f"x not equal, {x_float, x_, x}"
                # assert abs(y_float - y_) <= 0.001, f"y not equal, {y_float, y_, y}"
                corneal_surface[angle][mire_num] = (x_, y_, z)
                plot_x.append(x_)
                plot_y.append(y_)
                plot_z.append(z)
        plot_x.append(0.0), plot_y.append(0.0), plot_z.append(0.0)

        plot_x = np.asarray(plot_x)
        plot_y = np.asarray(plot_y) 
        plot_z = np.asarray(plot_z)

        num_points = len(plot_x)
        plot_x = np.array(plot_x).reshape((num_points, 1))
        plot_y = np.array(plot_y).reshape((num_points, 1))
        plot_z = np.array(plot_z).reshape((num_points, 1))

        idx = plot_z<2.0

        plot_x = plot_x[idx]
        plot_y = plot_y[idx]
        plot_z = plot_z[idx]

        rho = np.sqrt(np.square(plot_x) + np.square(plot_y))
        xy_norm = rho.max()

        plot_x, plot_y = plot_x / xy_norm, plot_y / xy_norm
        max_radius = (max_radius // 2) * 2

        return plot_x, plot_y, plot_z, xy_norm, max_radius, relative_points