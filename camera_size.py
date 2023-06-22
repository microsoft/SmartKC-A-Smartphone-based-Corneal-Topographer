from hardware.hardware_constants import HardwareConstants

def get_slope(obj_dim_pix, img_dims, dist, sensor_dims=(6.4, 4.8), f_len=4.76):
    obj_dim_img = (
        obj_dim_pix[0] * sensor_dims[0] / img_dims[0],
        obj_dim_pix[1] * sensor_dims[1] / img_dims[1],
    )
    obj_dim_size = (obj_dim_img[0] ** 2 + obj_dim_img[1] ** 2) ** 0.5
    # print("obj_dim_size", obj_dim_size)
    obj_slope = obj_dim_size / f_len
    obj_dim_real = dist * obj_dim_img[0] / f_len, dist * obj_dim_img[1] / f_len
    return obj_slope


def get_arc_step_params(
    pixels_size, w, h, sensor_dims, f_len, working_distance, model_file, mid_point=False
):

    f = open(model_file)
    model = []
    if mid_point == True:
        model_temp = []
        for idx, line in enumerate(f):
            # NOTE: skip the first mire (since that is black)
            if idx == 0:
                continue
            line = line.strip().split()
            model_temp.append((float(line[0]), float(line[1])))
        for idx in range(len(model_temp) - 1):
            model.append(
                (   
                    # uncomment and use below in-case of oldest placido head
                    (model_temp[idx][0] + model_temp[idx + 1][0] - HardwareConstants.placido_thickness_mm + HardwareConstants.thickness_adjustment_mm) / 2.0, 
                    #(model_temp[idx][0] + model_temp[idx + 1][0]) / 2.0,
                    (model_temp[idx][1] + model_temp[idx + 1][1]) / 2.0,
                )
            )
            # (model_temp[idx][1]+model_temp[idx+1][1]-2)/2.0))
    else:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            line = line.strip().split(" ")
            model.append((float(line[0]), float(line[1])))
    f.close()

    ys, zs, oz, oy, k = [], [], [], [], []
    for idx, (ring_r, ring_pos) in enumerate(model):
        if idx >= len(pixels_size):
            break
        y_s, z_s = ring_r, -working_distance + ring_pos
        slope = get_slope(
            (pixels_size[idx][0], pixels_size[idx][1]),
            (w, h),
            1,
            sensor_dims=sensor_dims,
            f_len=f_len,
        )
        k.append(slope)
        oz.append(z_s)
        oy.append(y_s)

    k, oz, oy = [k[0]] + k, [oz[0]] + oz, [oy[0]] + oy

    return k, oz, oy
