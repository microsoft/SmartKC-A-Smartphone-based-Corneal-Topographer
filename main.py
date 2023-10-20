#!/usr/bin/env python
# Author: Siddhartha Gairola (t-sigai at microsoft dot com))

import os
import traceback
import cv2
import logging
from datetime import date
import numpy as np
import matplotlib.ticker as ticker
import pandas as pd

from mire_detection_dl import detect_mires_from_mask
from arc_step_method import arc_step_method
from segment_mires import mire_segmentation

np.set_printoptions(threshold=np.inf)
import argparse
import csv

import warnings
warnings.filterwarnings("ignore")

# external modules
from preprocess import preprocess_image
from mire_detection import detect_mires_img_proc, clean_points
from get_maps import *
from utils import *
from metrics import *
from zernike_smartkc import RZern
from mire_detection_graph import detect_mires_from_graph
from constants import Constants

# command line arguments (if any)
parser = argparse.ArgumentParser(description="KT Processing Pipeline")
parser.add_argument(
    "--start_angle", 
    default=0, 
    type=float, 
    help="Starting meridian",
)
parser.add_argument("--end_angle", 
    default=360, 
    type=float, 
    help="Ending Meridian",
)
parser.add_argument("--jump", 
    default=1, 
    type=int, 
    help="Jump between meridians",
)
parser.add_argument(
    "--n_mires", 
    default=22, 
    type=int, 
    help="Number of mires to process",
)

# TODO - Change to placido_length
parser.add_argument(
    "--working_distance",
    default=75.0,
    type=float,
    help="Distance of cone end from cornea",
)
# TODO - Add argument to input total working distance : argname : working_distance

parser.add_argument(
    "--camera_params",
    default=None,
    type=str,
    help="Camera parameters: sensor dimensions (width x height), focal length (space separated string)",
)
parser.add_argument(
    "--model_file",
    default=None,
    type=str,
    help="File with details about the placido head model",
)
parser.add_argument(
    "--base_dir", 
    default="images", 
    type=str, 
    help="Image data directory",
)
parser.add_argument(
    "--image_name", 
    default=None, 
    type=str, 
    help="Test input image.",
)

parser.add_argument(
    "--gap2",
    default=4,
    type=float,
    help="Accounting for gap (in mm) between camera pupil and smallest ring.",
)
parser.add_argument(
    "--center_selection",
    default="default",
    type=str,
    help="Flag for setting mode for center selection (auto or manual-pc or manual-app)",
)
parser.add_argument(
    "--centers_filename",
    default=None,
    type=str,
    help="Filename to read the centers from in format: image_name center_x center_y",
)
parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    help="Output directory name. If not provided, the current date is used for the directory name.",
)

parser.add_argument(
    "--zernike_degree",
    default=8,
    type=int,
    help="Degree of the zernike polynomial used to fit the corneal surface"
)

parser.add_argument(
    "--mire_seg_method",
    choices=Constants.MIRE_SEGMENTATION_METHODS,
    required=True,
    help="Method used to segment mires",
)

parser.add_argument(
    "--mire_loc_method",
    choices=Constants.MIRE_LOCALIZATION_METHODS,
    required=True,
    help="Method used to locate mire points",
)

parser.add_argument(
    "--verbose",
    action="store_true",
    default=False,
    help="Flag to enable verbose logging",
)

def plot_and_save_corneal_surface(x, y, z, output, save = True):
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    df.to_csv(output + "corneal_surface.csv", index=False)
    plt.figure()
    ax = plt.axes(projection = '3d')
    ax.scatter3D(x, y, z)
    plt.savefig(output + "corneal_surface_3d.png")
    plt.close()

class corneal_top_gen:

    def __init__(
        self, model_file, working_distance, sensor_dims, f_len, start_angle, end_angle, jump, upsample, n_mires, f_gap1, test_name, zernike_degree=[8]
        ):
        self.model_file = model_file # file which consists of the placido head dimensions
        self.working_distance = working_distance # distance between camera pupil and cornea apex
        self.sensor_dims = sensor_dims # width x height of camera sensor
        self.f_len = f_len # focal length of camera
        self.f_gap1 = f_gap1 # function which maps 1/mire_21_radius to gap1
        self.start_angle = start_angle # start meridian
        self.end_angle = end_angle # end meridian
        self.jump = jump # diff between the consec angles when processing mires
        self.ups = upsample # if the image has to be upsampled or not
        self.n_mires = n_mires # number of mires to process
        self.zernike_degree = zernike_degree # degree of the zernike polynomial used for fitting
        self.output = test_name
    
    # to handle different phones with different intrinsic camera params
    # TODO: Replace f_gap1_wrapper with f_gap1 in the code below
    def f_gap1_wrapper(self, f_gap1, mire_radius, base_focal_length, base_res_width, base_sensor_width, current_res_width):
        f_base_f_curr = base_focal_length/self.sensor_dims[2]
        base_r_w_base_s_w = base_res_width/base_sensor_width
        mire_radius = (mire_radius/current_res_width*self.sensor_dims[0])*f_base_f_curr*base_r_w_base_s_w
        mire_radius = mire_radius*2.0 # since original image was 6000x8000 at the time of calibration in simulation
        return round(f_gap1(1/mire_radius), 2)

    def zernike_smoothening(self, image_name, plot_x, plot_y, plot_z, 
        xy_norm, xv, yv, max_r, relative_points):

        error = -1
        for zern_deg in self.zernike_degree:
            zern_deg = int(zern_deg)

            # zernike fitting takes place, c1: zernike coefficients
            cart = RZern(zern_deg)
            cart.make_cart_grid(plot_x, plot_y, scale_by=xy_norm)
            c1 = cart.fit_cart_grid(plot_z)[0]

            # for grid xv, yv
            cart = RZern(zern_deg)
            cart.make_cart_grid(xv, yv, scale_by=xy_norm)
            Phi = cart.eval_grid(c1, matrix=True)
            Phi = Phi[np.isfinite(Phi)].max() - Phi
            rho = np.abs(np.sqrt(xv ** 2 + yv ** 2) * xy_norm)

            # compute curvatures k1 (instantaneous) and k2 (axial)
            k1, k2 = cart.eval_curvature_grid(c1, matrix=True)
            k1, k2 = abs(k1), abs(k2)
            k1_raw = k1.copy()
            inst_roc, axial_roc = 1 / k1, 1 / k2 # computing roc from curvatures

            '''
            # older simK computation
            check0 = np.isfinite(inst_roc) * (rho<=xy_norm*0.7) # getting the central 70% region
            error = (np.abs(inst_roc[check0]-7.8)/7.8).mean()*100 # computing error w.r.t a normal eye
            check0 = np.isfinite(inst_roc) * (rho <= 1.5) # getting only points within 3 mm diameter
            
            # find k1 angle
            angle_k1 = np.argwhere(k1[check0].max() == k1)[0]
            angle_k1 = np.arctan(
                (angle_k1[0] - k1.shape[0] // 2) / (angle_k1[1] - k1.shape[1] // 2 + 1e-9)
            )
            angle = round(-angle_k1 * 180 / np.pi, 0)
            k1, k2 = k1[check0], k2[check0]

            sim_k1 = round(337.5 * k1.max(), 2)
            sim_k2 = round(337.5 * k2[np.argmax(k1)], 2)
            average_k, diff = round((sim_k1 + sim_k2) / 2.0, 2), round(sim_k1 - sim_k2, 2)
            '''

            check = np.isnan(inst_roc); inst_roc[check] = 1e6;
            check = np.isnan(axial_roc); axial_roc[check] = 1e6;

            tan_map = generate_tan_map(
                inst_roc,
                gt_pal,
                gt_r,
                (inst_roc.shape[1] // 2, inst_roc.shape[0] // 2),
                max_r,
                None
                #str(err1) + "_" + str(err2) + "_" + str(zern_deg) + "_" + str(jump) + "_" + image_name,
                #output_folder=self.output + "/" + image_name,
            )

            # generate axial map using the meridonial averaging method
            axial_map, k2 = generate_axial_map(
                1 / inst_roc,
                gt_pal,
                gt_p,
                (inst_roc.shape[1] // 2, inst_roc.shape[0] // 2),
                max_r,
                None
                #str(err1) + "_" + str(zern_deg) + "_" + str(jump) + "_" + image_name,
                #output_folder=self.output + "/" + image_name,
            )
            
            k2_raw = k2.copy()
            '''
            # old simK computation
            # re-computed after generating axial map using the averaging method
            k2_raw = k2.copy()
            k2 = k2[check0]
            sim_k2 = round(337.5 * k2[np.argmax(k1)], 2)
            average_k, diff = round((sim_k1 + sim_k2) / 2.0, 2), round(sim_k1 - sim_k2, 2)
            '''

            # draw the 3mm, 5mm, 7mm circles
            r_1 = int(float(max_r)/xy_norm*0.5)
            r_2 = int(float(max_r)/xy_norm*1.0)
            r_3 = int(float(max_r)/xy_norm*1.5)
            r_3_5 = int(float(max_r)/xy_norm*1.75)
            r_5 = int(float(max_r)/xy_norm*2.5)
            r_7 = int(float(max_r)/xy_norm*3.5)

            # new simK computation
            sim_k2, sim_k1, _, angle_k1 = compute_simk(k2.copy(), (k2.shape[1]//2, k2.shape[0]//2), r_3)
            average_k, diff = round((sim_k1 + sim_k2) / 2.0, 2), round(sim_k1 - sim_k2, 2)
            angle_k1 *= np.pi/180

            tan_map = draw_circles(
                tan_map,
                (inst_roc.shape[1] // 2, inst_roc.shape[0] // 2),
                [r_3, r_5, r_7],
                angle_k1,
                (sim_k1, sim_k2)
            )

            axial_map = draw_circles(
                axial_map,
                (inst_roc.shape[1] // 2, inst_roc.shape[0] // 2),
                [r_3, r_5, r_7],
                angle_k1,
                (sim_k1, sim_k2)
            )

            # compute CLMI & PPK score
            ppk, _, _, _ = clmi_ppk(
                k2_raw.copy(),
                axial_map.copy(),
                r_2,
                r_7,
                (inst_roc.shape[1] // 2, inst_roc.shape[0] // 2),
            )

            # compute KISA score
            # KISA(
            #     k1_raw.copy(),
            #     (k1_raw.shape[1] // 2, k1_raw.shape[0] // 2),
            #     relative_points,
            #     r_3,
            #     diff,
            # )

            #KISA(k2_raw, (k2_raw.shape[1]//2, k2_raw.shape[0]//2), relative_points, r_3, diff)
            #compute_tilt_factor(k1_raw.copy(), tan_map.copy(), r_1, r_3_5, (k1_raw.shape[1]//2, k1_raw.shape[0]//2), angle_k1, image_name)
            compute_tilt_factor(k2_raw.copy(), axial_map.copy(), r_1, r_3_5, 
                (k1_raw.shape[1]//2, k1_raw.shape[0]//2), angle_k1, image_name, output_folder=self.output)

        return error, tan_map, axial_map, sim_k1, sim_k2, round(-angle_k1 * 180 / np.pi, 1), average_k, diff, ppk


    # main runner function to generate topography maps from input image
    def generate_topography_maps(
        self, base_dir, image_name, mire_seg_method, mire_loc_method, crop_dims=(1200,1200), iso_dims=500, 
        center=(-1, -1), upsample=None,
        err2=0, skip_angles=[[-1, -1], [-1, -1]],
        center_selection="auto",
        marked_center = None,
    ):
        self.skip_angles = skip_angles
  
        # create output directory if not present
        if not os.path.isdir(self.output):
            os.mkdir(self.output)

        # create directory to store output
        if not (os.path.isdir(self.output+"/"+image_name.split(".jpg")[0])):
            os.mkdir(self.output+"/"+image_name.split(".jpg")[0])
        

        # Step 1: Image Centering and Cropping
        # Step 2: Image Enhancement, Cleaning & Enhancement
        # Step 3: Locate Image Center
        # This can be done in 3 ways:
        # 1 Using center of central mire, or
        # 2 Centroid of Segmented Image (compute it's center of mass)
        # 3 User selects center manually if center = (-1, -1)
        image_gray, center = preprocess_image(
            base_dir,
            image_name,
            center,
            crop_dims=crop_dims,
            iso_dims=iso_dims,
            output_folder=self.output,
            center_selection=center_selection,
            marked_center=marked_center
        )

        mire_seg = mire_segmentation(mire_seg_method, center, Constants.DL_MODEL_FILE)
        image_seg, image_edge = mire_seg.segment_mires(image_gray)
            # upsample image to higher resolution
        if mire_seg_method == Constants.IMG_PROC_MIRE_SEG and self.ups > 1:
            image_gray, image_seg, image_edge, center = increase_res(
                image_gray, image_seg, image_edge, center, self.ups, image_name.split(".jpg")[0]
            )

        image_name = image_name.split(".jpg")[0]
        cv2.imwrite(self.output + "/" + image_name + "/" + image_name + "_seg.png", convert_to_binary(image_seg))

        # Step 4: Mire detection + detect meridinial points on respective mires
        if mire_loc_method == Constants.RADIAL_SCAN_LOC_METHOD:
            if mire_seg_method == Constants.IMG_PROC_MIRE_SEG:
                image_cent_list, center, others = detect_mires_img_proc(
                    image_seg, image_gray, center, self.jump, self.start_angle, self.end_angle
                )
            elif mire_seg_method == Constants.DL_MIRE_SEG:
                image_cent_list, image_mp = detect_mires_from_mask(
                    image_seg,
                    center,
                    self.n_mires,
                    np.dstack((image_gray, np.dstack((image_gray, image_gray))))
                )
            else:
                raise ValueError("Invalid mire segmentation method")
        elif mire_loc_method == Constants.GRAPH_CLUSTER_LOC_METHOD:
            if mire_seg_method == Constants.DL_MIRE_SEG:
                image_seg = convert_to_binary(image_seg)
            image_cent_list, image_mp = detect_mires_from_graph().fetch_mire_points(image_gray, image_seg, center, self.n_mires, self.start_angle, self.end_angle, self.jump, self.output + "/" + image_name + "/")            
        else:
            raise ValueError("Invalid mire localization method")

        # image_name = image_name.split(".jpg")[0]

        # cv2.imwrite(self.output+"/" + image_name + "/" + image_name + "_mp.png", image_mp)

        # clean points
        # TODO - Check if heuristics are needed for traditional Img Proc
        r_pixels, flagged_points, coords, image_mp = clean_points(
            image_cent_list, image_gray.copy(), image_name, center, mire_loc_method, self.n_mires, self.jump, self.start_angle, self.end_angle, 
            output_folder=self.output, 
        )

        mire_20_radii = [r_pixels[20][i] for i in range(self.start_angle, self.end_angle, self.jump) if (20, i) not in flagged_points]

        max_r = np.nanmax(mire_20_radii)
        min_r = np.nanmin(mire_20_radii)
        mire_20_radius = (2*max_r + min_r)/3.0 * 2
        print(f"Mire 20 radius - {mire_20_radius}")

        # mire_20_radius = np.nanmean(r_pixels[20][15:330])*2.0
        if self.f_gap1 is not None:
            err1 = self.f_gap1(1/mire_20_radius)

        # get image real dimensions, account for upsampling
        h, w = cv2.imread(base_dir + "/" + image_name + ".jpg").shape[:2]
        h, w = self.ups * h, self.ups * w 
        
        errors = []
        # Steps 5, 6 & 7
        logging.info(f"Effective Working distance = {self.working_distance + err1 + err2}")                
        if mire_loc_method == Constants.RADIAL_SCAN_LOC_METHOD:
            assert flagged_points == [], "Flagged points not empty"
        arc_step = arc_step_method(self.model_file, self.start_angle, self.end_angle, self.jump, self.skip_angles)
        x, y, z, xy_norm, max_radius, relative_points= arc_step.run( 
            image_seg, image_name, center, r_pixels, coords, h, w, self.working_distance, self.sensor_dims, self.f_len, err1, err2, flagged_points, self.n_mires - 1)

        plot_and_save_corneal_surface(x,y,z,self.output + "/" + image_name + '/', save = True)

        ddx = np.linspace(-1.0, 1.0, int(2 * max_radius))
        ddy = np.linspace(-1.0, 1.0, int(2 * max_radius))
        xv, yv = np.meshgrid(ddx, ddy)

        logging.info("Arc-step complete, running zernike smoothening")
        error, tan_map, axial_map, sim_k1, sim_k2, angle, average_k, diff, ppk = self.zernike_smoothening(
            image_name, x, y, z, xy_norm, xv, yv, max_radius, relative_points)                    
        sims = [sim_k1, sim_k2, angle, average_k, diff, ppk]

        # overlay on gray image
        image_overlay = np.dstack((image_gray, np.dstack((image_gray, image_gray)))).astype(np.uint8)
        temp_map = np.zeros_like(image_overlay)
        # get tangential map overlay
        # TODO - Figure out why a (% 2) is needed here
        temp_map[
            center[1] - tan_map.shape[0] // 2 : center[1] + tan_map.shape[0] // 2 + tan_map.shape[0] % 2,
            center[0] - tan_map.shape[1] // 2 : center[0] + tan_map.shape[1] // 2 + tan_map.shape[1] % 2,
            :] = tan_map
        tan_map = temp_map.copy()
        # get axial map overlay
        temp_map = np.zeros_like(image_overlay)
        temp_map[
            center[1] - axial_map.shape[0] // 2 : center[1] + axial_map.shape[0] // 2 + axial_map.shape[0] % 2,
            center[0] - axial_map.shape[1] // 2 : center[0] + axial_map.shape[1] // 2 + axial_map.shape[1] % 2,
            :] = axial_map
        axial_map = temp_map.copy()

        mask = axial_map[:, :, 0] > 0
        image_overlay[mask] = [0, 0, 0]
        tan_map_overlay = image_overlay + tan_map
        axial_map_overlay = image_overlay + axial_map

        with open(self.output + f"/{mire_seg_method}_{mire_loc_method}_simk.csv", "a") as f:
            f.write(image_name + "," + str(sims[0]) + "," + str(sims[1]) + "," + str(sims[2]) + "," + str(self.working_distance + err1 + err2) + "\n")

        cv2.putText(
            tan_map_overlay,
            "Sim K1: "+ str(sims[0])+ "D @"+ str(sims[2])+ " K2: "+ str(sims[1])+ "D @"+ str(sims[2] + 90),
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.putText(tan_map_overlay,
            "Avg: " + str(sims[3]) + "D Diff: " + str(sims[4]) + "D",
            (5, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            axial_map_overlay,
            "Sim K1: "+ str(sims[0]) + "D @" + str(sims[2]) + " K2: "+ str(sims[1])+ "D @"+ str(sims[2] + 90),
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            axial_map_overlay,
            "Avg: " + str(sims[3]) + "D Diff: " + str(sims[4]) + "D",
            (5, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    
        cv2.imwrite(
            self.output+"/" + image_name + "/" + image_name + "_tan_map_overlay.png",
            tan_map_overlay,
        )
        cv2.imwrite(
            self.output+"/" + image_name + "/" + image_name + "_axial_map_overlay.png",
            axial_map_overlay,
        )
    
        logging.warning("Test Complete!")
        return errors, sims, [image_gray, image_seg, image_mp, tan_map_overlay, axial_map_overlay]

def read_center(center_filename, image_name):
    file = open(center_filename, 'r')
    lines = file.readlines()
    for line in lines:
        if (line.split()[0]+".jpg") == image_name:
            return [int(line.split()[1]), int(line.split()[2])]
    return (-1, -1)

if __name__ == "__main__":
    # parsing arguments
    args = parser.parse_args()
    # Default config - WARNING
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # getting parameters for corneal_top_obj
    f_inv_20 = None
    if args.gap2 == 3:
        f_inv_20 = np.poly1d([3652.09954861, -17.22770463]) # 3 mm gap2, mire_21, id_20
    elif args.gap2 == 4:
        f_inv_20 = np.poly1d([3617.81645183, -17.2737687]) # 4 mm gap2, mire_21, id_20
    elif args.gap2 == 5:
        f_inv_20 = np.poly1d([3583.52156815, -17.31674123]) # 5 mm gap2, mire_21, id_20

    # fetch camera parameters
    sensor_dims = None
    f_len = None
    if (args.camera_params is not None):
        sensor_dims = (
            float(args.camera_params.split()[0]),
            float(args.camera_params.split()[1]),
        )  # "4.27, 5.68, 4.25"
        f_len = float(args.camera_params.split()[2]) # focal length of the camera

    # get details for current test image
    base_dir = args.base_dir  # base directory
    skip_angles = [[-1, -1], [-1, -1]]
    center = (-1, -1)

    # call function to run pipeline and generate_topography_maps
    # expects image to be in .jpg format
    to_process = list(filter(lambda name: name.endswith('.jpg'), os.listdir(base_dir)))
    failed = set()
    execution_order = []
    
    # determine execution order from center_selection
    center_selection = args.center_selection
    if (center_selection == 'default'): execution_order = ['manual-android', 'auto', 'manual-pc']
    elif (center_selection == 'manual-android'): execution_order = ['manual-android', 'auto', 'manual-pc']
    elif (center_selection == 'auto'): execution_order = ['auto', 'manual-pc']
    elif (center_selection == 'manual-pc'): execution_order = ['manual-pc']

    # set the output directory
    if args.output_dir is None:
        output_dir = date.today().strftime("%d_%m_%Y")
    else:
        output_dir = args.output_dir
    
    for selection_mode in execution_order:
        while len(to_process):
            filename = to_process.pop()
            print("Running for file:", filename, "with mode:", selection_mode)
            try:
                csv_file_parts = filename.split('_')[:-2]
                csv_file_name = '_'.join(csv_file_parts) + '.csv'
                csv_file_path = base_dir + '/' + csv_file_name
                
                focal_length = None
                marked_center = None
                
                # Open only if csv available
                if (os.path.exists(csv_file_path)):
                    # try to read values
                    with open(csv_file_path, newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            if row["image_name"] == filename:
                                f_len = row['focal_length']
                                sensor_dims = list(map(float, row['camera_physical_size'].split('x')))
                                sensor_dims.sort()
                                sensor_dims = tuple(sensor_dims)
                                marked_center = list(map(float, row['marked_center'].split('|')))
                                break
                
                if args.centers_filename is not None:
                    center = read_center(base_dir+args.centers_filename, filename)

                # create the corneal_top_gen class object
                corneal_top_obj = corneal_top_gen(
                    args.model_file, args.working_distance, sensor_dims, 
                    f_len, args.start_angle, args.end_angle, args.jump, 
                    Constants.IMG_PROC_SEG_PARAMS["UPSAMPLE"], args.n_mires, f_inv_20, output_dir, zernike_degree=[args.zernike_degree],
                    )
                
                # TODO: Clean up such that only one center is passed
                corneal_top_obj.generate_topography_maps(
                base_dir,
                filename,
                args.mire_seg_method,
                args.mire_loc_method,
                center=center,
                err2=args.gap2,
                center_selection=selection_mode,
                marked_center=marked_center,
                )
            except Exception as e:
                traceback.print_exc()
                failed.add(filename)
        print("Following files failed for center mode", selection_mode, " : ", failed)
        # Try failed files for next mode
        to_process = list(failed)
        failed = []
    
    if (len(failed)):
        print("Failed to generate heatmaps for files: ", failed)