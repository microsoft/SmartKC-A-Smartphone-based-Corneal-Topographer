import numpy as np
class Constants:
    UNKNOWN_RADIUS = np.nan
    UNKNOWN_COORDINATE = (np.nan, np.nan)

    IMG_PROC_MIRE_SEG = "img_proc"
    DL_MIRE_SEG = "dl"
    MIRE_SEGMENTATION_METHODS = [IMG_PROC_MIRE_SEG, DL_MIRE_SEG]

    RADIAL_SCAN_LOC_METHOD = "radial_scan"
    GRAPH_CLUSTER_LOC_METHOD = "graph_cluster"
    MIRE_LOCALIZATION_METHODS = [RADIAL_SCAN_LOC_METHOD, GRAPH_CLUSTER_LOC_METHOD]

    IMG_PROC_SEG_PARAMS = {
        "UPSAMPLE" : 1,
        "DOWNSAMPLE" : True,
        "BLUR" : True
    }
    DL_MODEL_FILE = "./get_center/segment_and_get_center_epoch_557_iter_14.pkl"
    GRAPH_CLUSTER_MIN_CONNECTED_COMPONENT_SIZE = 10

    POSTPROCESS_MASKING_THRESHOLD = 20
    MASK_LENGTH = 10
