"""
1. Extract DL mire-mask and convert to binary
2. Run radial scanning on DL-based binary mask
3. Construct graph from mire points - edges are added between 2 points which are close 
enough in the tangential (tolerance = 2.r.theta) and normal direction (smaller tolerance). 
(Distances between 2 points are projected into tangential and normal directions - 
normal direction tolerance is stricter than tangeintial tolerance)
(One point can also have edges to more than 2 neighbouring points, which satisfy the tolerance condition)
4. Find connected components in this graph
5. If multiple mire-ids (obtained from Step-2) are found in the same connected component, 
then (a) Overwrite mire-ids to mode of the current connected component
    (b) shift the mire-ids in the subsequent mires for the corresponding angles (by moving along the radial ray)
6. Due to propagating shifts, there might be multiple mire-ids in the same angle. All such points are removed.

"""
import numpy as np
from utils import process_mires, plot_color_rb
import pandas as pd
import cv2
from constants import Constants

class detect_mires_from_graph:
    def radial_scanning(self, image_seg, image_orig, center, 
        jump=2, start_angle=0, end_angle=360):

        image_cent_list = []
        image_inv = 255 - image_seg
        image_mp = np.dstack((image_orig, np.dstack((image_orig, image_orig))))
        for angle in np.arange(start_angle, end_angle, jump):
            cent, line = process_mires(image_seg.copy(), center, angle, weights=None)
            cent_inv, _ = process_mires(image_inv.copy(), center, angle, weights=None)
            cent = cent + cent_inv
            cent.sort(key = lambda x: x[2])
            image_cent_list.append(cent)

            image_mp = plot_color_rb(image_mp, cent)

        return image_cent_list, image_mp
    
    def construct_graph(self, nodes, center, radial_tolerance):
        edges = [[] for _ in nodes]
        for i, node in enumerate(nodes):
            radius = node[2]
            curr_node = node[3:5]
            unmatched_nodes = nodes[:,3:5]

            radial_vector = (curr_node - center)
            radial_vector /= np.linalg.norm(radial_vector)

            tangential_vector = np.array([-radial_vector[1], radial_vector[0]])
            # Min 1.5 tolerance so that diagonally 
            tangential_tolerance = max(2 * radius * np.pi / 180, 1.5)

            unmatched_nodes = unmatched_nodes - curr_node
            tangential_distances = np.abs(np.dot(unmatched_nodes, tangential_vector))        
            normal_distances = np.abs(np.dot(unmatched_nodes, radial_vector))

            dist_mask = (tangential_distances <= tangential_tolerance) & (normal_distances <= radial_tolerance) & (nodes[:, 1] != node[1])        
            new_nodes = np.nonzero((dist_mask))

            for new_node in new_nodes[0].tolist():
                if new_node == i:
                    continue
                if new_node not in edges[i]:
                    edges[i].append(new_node)
                if i not in edges[new_node]:
                    edges[new_node].append(i)
        return edges


    def find_connected_components(self, data : pd.DataFrame, center, radial_tolerance = 1):
        nodes = data.values
        marked_nodes = np.ones((len(nodes))) * -1
        assert len(marked_nodes) == len(nodes)    

        edges = self.construct_graph(nodes, center, radial_tolerance)

        components = np.ones((len(nodes))) * -1
        curr_idx = 1
        for i,node in enumerate(nodes):
            if components[i] != -1:
                continue
            current_component = [i]
            current_component_angles = [node[1]]
            while len(current_component) != 0:
                idx = current_component.pop()
                if components[idx] != -1:
                    continue
                
                components[idx] = curr_idx
                new_nodes = [e for e in edges[idx] if components[e] == -1 and e not in current_component and nodes[e][1] not in current_component_angles]
                current_component.extend(new_nodes)
                current_component_angles.extend([nodes[e][1] for e in new_nodes])

            curr_idx += 1
        data["component"] = components    
        return data

    def correct_mire_shifts(self, data : pd.DataFrame):
        data.sort_values(by=["radius", "mire_num", "angle"], inplace=True)

        components = data["component"].unique()
        for c in components:
            subset = data[data["component"] == c]

            # Removing components with less than 10 points
            if len(subset) < Constants.GRAPH_CLUSTER_MIN_CONNECTED_COMPONENT_SIZE:
                data.loc[data["component"] == c, "component"] = -1

            mire_counts = dict(subset["mire_num"].value_counts())
            num_points = len(subset)
            max_perc = max(mire_counts.values()) / num_points
            if max_perc != 1:
                mires_in_component = dict(sorted(mire_counts.items(), key=lambda x: x[1], reverse=True))
                for idx, (mire_num, count) in enumerate(mires_in_component.items()):
                    if idx == 0:
                        current_mire = mire_num
                        continue
                    else:
                        diff = current_mire - mire_num
                        angles = subset[subset["mire_num"] == mire_num]["angle"].values
                        data.loc[(data["mire_num"] >= mire_num) & (data["angle"].isin(angles)), "mire_num"] += diff
        data = data[data["component"] >= 0]
        return data

    def remove_repeated_mire_nums(self, data: pd.DataFrame):
        mires = sorted(data["mire_num"].unique())
        for mire_num in mires:
            mire = data[data["mire_num"] == mire_num]
            mire = mire.sort_values(by="angle")
            angles = mire["angle"].value_counts()

            angles = angles[angles > 1].index.tolist()

            data.loc[(data["mire_num"] == mire_num) & (data["angle"].isin(angles)), "component"] = -1

        data = data[data["component"] >= 0]
        return data

    def visualize_mire_locations(self, data : dict, img, out_dir):
        colors = np.random.randint(0, 255, (len(data), 3), )
        for angle, mires in enumerate(data):
            for mire_num, (x,y,radius) in enumerate(mires):
                img[int(y), int(x)] = colors[mire_num]
        cv2.imwrite(out_dir + "image_mp.jpg", img)


    def fetch_mire_points(self, img, mire_mask, center, n_mires, start_angle, end_angle, jump, out_dir):
        # Step-2
        radii_list, image_mp = self.radial_scanning(mire_mask, img, center, start_angle=start_angle, end_angle=end_angle, jump=jump)
        data = []

        # Converting to dataframe
        for angle, r_list in enumerate(radii_list):
            for mire_num, radius in enumerate(r_list):
                if mire_num >= n_mires:
                    break
                data.append([mire_num, angle, radius[2], radius[0], radius[1]])
        
        data = np.asarray(data)
        data = pd.DataFrame(data, columns=["mire_num", "angle", "radius", "x", "y"])

        # Step-3 and 4
        data = self.find_connected_components(data, center)
        # Step-5
        # Correcting shifts in mires within a connected component 
        data = self.correct_mire_shifts(data)
        # Step-6
        # Removing mires that are repeated in the same angle
        data = self.remove_repeated_mire_nums(data)

        image_cent_list = [[] for _ in range(360)]

        # Converting the data to a list of mires,angles, (x,y,radius)
        data.sort_values(by=["mire_num", "angle"], inplace=True)
        for angle in np.arange(start_angle, end_angle, jump):
            angle_points = data[data["angle"] == angle]
            for mire_num in range(n_mires):
                point = angle_points[angle_points["mire_num"] == mire_num]
                if len(point) == 0:
                    image_cent_list[angle].append([Constants.UNKNOWN_COORDINATE[0], Constants.UNKNOWN_COORDINATE[1], Constants.UNKNOWN_RADIUS])
                else:
                    radius = np.sqrt((point["x"].values[0]-250)**2 + (point["y"].values[0]-250)**2)
                    image_cent_list[angle].append((point["x"].values[0], point["y"].values[0], radius))

        img_color = np.dstack((img, np.dstack((img, img))))    

        return image_cent_list, image_mp