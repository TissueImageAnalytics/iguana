"""
Functions for feature extraction
"""

import cv2
import numpy as np
import pandas as pd
import uuid
import math

from shapely.geometry import Polygon
from scipy.spatial import distance, Voronoi, Delaunay
from sklearn.neighbors import KDTree

from .bam_utils import get_enclosing_ellipse, ellipse_to_circle, apply_transform, best_alignment_metric


def get_contour_feats(cnt):
    """Get morphological features from input contours."""

    # calculate some things useful later:
    m = cv2.moments(cnt)

    # ** regionprops **
    Area = m["m00"]
    if Area > 0:
        Perimeter = cv2.arcLength(cnt, True)
        # bounding box: x,y,width,height
        BoundingBox = cv2.boundingRect(cnt)
        # centroid    = m10/m00, m01/m00 (x,y)
        Centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"])

        # EquivDiameter: diameter of circle with same area as region
        EquivDiameter = np.sqrt(4 * Area / np.pi)
        # Extent: ratio of area of region to area of bounding box
        Extent = Area / (BoundingBox[2] * BoundingBox[3])

        # CONVEX HULL stuff
        # convex hull vertices
        ConvexHull = cv2.convexHull(cnt)
        ConvexArea = cv2.contourArea(ConvexHull)
        # Solidity := Area/ConvexArea
        Solidity = Area / ConvexArea

        # ELLIPSE - determine best-fitting ellipse.
        centre, axes, angle = cv2.fitEllipse(cnt)
        MAJ = np.argmax(axes)  # this is MAJor axis, 1 or 0
        MIN = 1 - MAJ  # 0 or 1, minor axis
        # Note: axes length is 2*radius in that dimension
        MajorAxisLength = axes[MAJ]
        MinorAxisLength = axes[MIN]
        Eccentricity = np.sqrt(1 - (axes[MIN] / axes[MAJ]) ** 2)
        Orientation = angle
        EllipseCentre = centre  # x,y

    else:
        Perimeter = 0
        EquivDiameter = 0
        Extent = 0
        ConvexArea = 0
        Solidity = 0
        MajorAxisLength = 0
        MinorAxisLength = 0
        Eccentricity = 0
        Orientation = 0
        EllipseCentre = [0, 0]

    return {
        "area": Area,
        "perimeter": Perimeter,
        "equiv-diameter": EquivDiameter,
        "extent": Extent,
        "convex-area": ConvexArea,
        "solidity": Solidity,
        "major-axis-length": MajorAxisLength,
        "minor-axis-length": MinorAxisLength,
        "eccentricity": Eccentricity,
        "orientation": Orientation,
        "ellipse-centre-x": EllipseCentre[0],
        "ellipse-centre-y": EllipseCentre[1],
    }
    

def grab_cnts(input_info, ds_factor):
    """Get the contours from the input dictionary and remove excessive coordinates.
    
    input_info (list): List of input dictionaries
    ds_factor (int): Factor for removing coordinates
    
    """
    output_dict = {}
    # first get the input dictionary for a single tissue region
    for tissue_idx, input_dict in input_info.items():
        tissue_dict = {}
        for inst_id, info in input_dict.items():
            cnt = info["contour"]
            cnt = cnt[::ds_factor, :]
            tissue_dict[inst_id] = cnt
        output_dict[tissue_idx] = tissue_dict
    
    return output_dict


def grab_centroids_type(input_info):
    """Get the contours from the input dictionary and remove excessive coordinates.
    
    input_info (list): List of input dictionaries
    
    """
    output_dict = {}
    # first get the input dictionary for a single tissue region
    for tissue_idx, input_dict in input_info.items():
        tissue_dict = {}
        for inst_id, info in input_dict.items():
            centroid = info["centroid"]
            type = info["type"]
            tissue_dict[inst_id] = {"centroid": centroid, "type": type}
        output_dict[tissue_idx] = tissue_dict
    
    return output_dict


def convert_to_df(input_dict, return_type=True):
    """Convert input dict to dataframe."""
    
    cx_list = []
    cy_list = []
    cnt_list = []
    type_list = []
    for info in input_dict.values():
        centroid = info["centroid"]
        cx_list.append(centroid[0])
        cy_list.append(centroid[1])
        cnt_list.append(info["contour"])
        if return_type:
            type_list.append(info["type"])
    
    if return_type:
        df = pd.DataFrame(data={"cx": cx_list, "cy": cy_list, "contour": cnt_list, "type": type_list})
    else:
        df = pd.DataFrame(data={"cx": cx_list, "cy": cy_list, "contour": cnt_list})

    return df



def filter_coords_msk(coords, mask, scale=1, mode="contour", label=None):
    """Filter input coordinates so that only coordinates within mask remain.
    
    Args:
        coords: input coordinates to filter
        mask: labelled tissue mask 
        scale: processing resolution to mask scale factor 
        mode: whether to check entire contour or jus the centroid 
    
    """

    unique_tissue = np.unique(mask).tolist()[1:]
    # populate empty dictionary - one per connected component in the tissue mask
    output_dict = {}
    for idx in unique_tissue:
        output_dict[idx] = {}

    # iterate over each object and check to see whether it is within the tissue
    for key, value in coords.items():
        # if a label is provided, then only consider the contour if it is equal to the label
        if label is not None and value["type"] != label:
            continue
        contours = value[mode]
        if mode == "centroid":
            contours = [contours]
        in_mask = False
        for coord in contours:
            coord = coord.astype("float64")
            coord *= scale
            coord = np.rint(coord).astype("int32")
            # make sure coordinate is within the mask
            if coord[0] > 0 and coord[1] > 0 and coord[0] < mask.shape[1] and coord[1] < mask.shape[0]:
                if np.sum(mask[coord[1], coord[0]]) > 0:
                    tissue_idx = int(mask[coord[1], coord[0]])
                    in_mask = True

        if in_mask:
            inst_uuid = uuid.uuid4().hex
            # add contour info to corresponding postion in output dictionary
            output_dict[tissue_idx][inst_uuid] = value
        
    return output_dict


def filter_coords_msk2(coords, mask1, mask2, scale=1, mode="centroid"):
    """Filter input coordinates so that only coordinates within mask remain.
    This function returns an additional dictionary to `filter_coords_msk()`,
    which only considers objects in a second provided mask.
    
    Args:
        coords: input coordinates to filter
        mask1: labelled tissue mask 
        mask2: binary mask for further sampling
        scale: processing resolution to mask scale factor 
        mode: whether to check entire contour or jus the centroid 
    
    Returns:
        output_dict1 (dict): dictionary of objects within mask
        output_dict2 (dict): subset of output_dict1 containing only objects that are also in mask2
        
    """

    unique_tissue = np.unique(mask1).tolist()[1:]
    # populate empty dictionary - one per connected component in the tissue mask
    output_dict = {}
    output_dict2 = {}
    for idx in unique_tissue:
        output_dict[idx] = {}
        output_dict2[idx] = {}

    # iterate over each object and check to see whether it is within the tissue
    for key, value in coords.items():

        contours = value[mode]
        if mode == "centroid":
            contours = [contours]
        in_mask = False
        in_mask2 = False
        for coord in contours:
            coord = coord.astype("float64")
            coord *= scale
            coord = np.rint(coord).astype("int32")
            # make sure coordinate is within the mask
            if coord[0] > 0 and coord[1] > 0 and coord[0] < mask1.shape[1] and coord[1] < mask1.shape[0]:
                if np.sum(mask1[coord[1], coord[0]]) > 0:
                    tissue_idx = int(mask1[coord[1], coord[0]])
                    in_mask = True
                    ######
                    if np.sum(mask2[coord[1], coord[0]]) > 0:
                        # only consider non-epithelilal classes!
                        if value["type"] != 2:
                            in_mask2 = True
        if in_mask:
            inst_uuid = uuid.uuid4().hex
            # add contour info to corresponding postion in output dictionary
            output_dict[tissue_idx][inst_uuid] = value
            if in_mask2:
                output_dict2[tissue_idx][inst_uuid] = value 


    return output_dict, output_dict2


def filter_coords_cnt(df, contour, mode="contour", return_type=True):
    """Filter input coordaintes so that only coordinates within contour remain.
    
    Args:
        df: input dataframe containing coordinates
        contour: contours for which to check
        mode: whether to check entire contour or jus the centroid 
    
    """

    assert mode in [
        "contour",
        "centroid",
    ], "`mode` must either be `contour` or `centroid`."

    output_dict = {}
    # iterate over each object and check to see whether it is within the contour (gland in this case)
    for idx, row in df.iterrows():
        if mode == "centroid":
            cnts = [[row["cx"], row["cy"]]]
        else:
            cnts = row["contour"]
        count = 0
        total = 0
        for cnt in cnts:
            total += 1
            cnt = np.rint(cnt).astype("int")
            contour = contour.astype("int")
            result = cv2.pointPolygonTest(contour, (int(cnt[0]), int(cnt[1])), False)
            # check if coordinate lies on or inside the gland contour
            if result != -1:
                count += 1
        # make sure at least 95% of contours are within / on the gland
        if count / total > 0.95:
            inst_uuid = uuid.uuid4().hex
            # add contour info to corresponding postion in output dictionary
            if return_type:
                output_dict[inst_uuid] = {
                    "centroid": [row["cx"], row["cy"]],
                    "contour": row["contour"],
                    "type": row["type"]
                    }
            else:
                output_dict[inst_uuid] = {
                    "centroid": [row["cx"], row["cy"]],
                    "contour": row["contour"]
                    }

    return output_dict


def points_list_pairwise_edt(list_a, list_b):
    """Get the parwise euclidean distance between two lists of contours.

    Args:
        list_a: first list of x,y contour coordinates, with shape Nx2.
        list_b: second list of x,y contour coordinates, with shape Nx2.

    Returns:
        pairwise euclidean distance.

    """
    pix_x_wrt_cnt_x = np.subtract.outer(list_a[:, 0], list_b[:, 0])  # INNER x CNT
    pix_y_wrt_cnt_y = np.subtract.outer(list_a[:, 1], list_b[:, 1])
    pix_x_wrt_cnt_x = pix_x_wrt_cnt_x.flatten()
    pix_y_wrt_cnt_y = pix_y_wrt_cnt_y.flatten()
    pix_cnt_dst = np.sqrt(pix_x_wrt_cnt_x ** 2 + pix_y_wrt_cnt_y ** 2)  # INNER x CNT
    pix_cnt_dst = np.reshape(pix_cnt_dst, (list_a.shape[0], list_b.shape[0]))
    return pix_cnt_dst


def get_centroid(cnt):
    """Get the centroid of a set of contour coordinates.

    Args:
        cnt: input contour coordinates.
    
    Returns:
        x and y centroid coordinates.

    """
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY


def get_boundary_distance(cnts, centroid):
    """"Get the euclidean distance of a centroid to the nearest contour boundary.
    
    Args:
        cnts: coordinates of contour
        centroid: coordinates of centroid
    
    """
    min_dst = 100000
    for cnt in cnts:
        dst = math.sqrt((cnt[0] - centroid[0]) ** 2 + (cnt[1] - centroid[1]) ** 2)
        if dst < min_dst:
            min_dst = dst

    return min_dst


def get_lumen_distance(lumen_info, centroid):
    """"Get the euclidean distance of a centroid to the nearest lumen boundary.
    
    Args:
        lumen_info: dict of lumen info
        centroid: coordinates of centroid
    
    """

    dst_list = []
    for _, info in lumen_info.items():
        cnts = info["contour"]
        min_dst = 100000
        for cnt in cnts:
            dst = math.sqrt((cnt[0] - centroid[0]) ** 2 + (cnt[1] - centroid[1]) ** 2)
            if dst < min_dst:
                min_dst = dst
        dst_list.append(min_dst)

    if len(dst_list) > 0:
        return min(dst_list)
    else:
        return np.nan


def get_voronoi_feats(coords):
    """Get voronoi diagram features."""
    vor = Voronoi(coords)
    regions = vor.regions
    vertices = vor.vertices
    
    area = []
    perim = []
    for region in regions:
        if len(region) > 0:
            if -1 in region:
                region.remove(-1)
            if len(region) > 2:
                # get the polygon area and perimeter info
                region_coords = vertices[region, :].astype('int')
                xy_coords = list(zip(region_coords[:, 0], region_coords[:, 1]))
                pgon = Polygon(xy_coords) # Assuming the OP's x,y coordinates
                area.append(pgon.area)
                perim.append(pgon.length)
            else:
                area.append(0)
                perim.append(0)

    return area, perim


def find_neighbors(pindex, triang):
    """Get the neighbouring vertices of a given vertex from Delaunay triangulation."""
    return triang.vertex_neighbor_vertices[1][triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex+1]]


def get_delaunay_feats(coords):
    """Get Delaunay triangulation features."""
    tess = Delaunay(coords)
    simps = tess.simplices

    area = []
    perim = []
    min_dst = []
    max_dst = []
    max_min_dst = []
    for idx, simp in enumerate(simps):

        tri_coords = coords[simp, :]
        # just being sure that 3 coordinates exist!
        if tri_coords.shape[0] == 3:
            # get the edge distance info
            dst_list = []
            for idx in range(3):
                if idx+1 == 3:
                    idx2 = 0
                else:
                    idx2 = idx+1
                a = tri_coords[idx, :]
                b = tri_coords[idx2, :]
                dst_list.append(distance.euclidean(a, b))
            min_dst.append(min(dst_list))
            max_dst.append(max(dst_list))
            max_min_dst.append(max(dst_list) / min(dst_list))

            # get the polygon area and perimeter info
            xy_coords = list(zip(tri_coords[:, 0], tri_coords[:, 1]))
            pgon = Polygon(xy_coords) # Assuming the OP's x,y coordinates
            area.append(pgon.area)
            perim.append(pgon.length)

    return area, perim, min_dst, max_dst, max_min_dst


def get_k_nearest_from_contour(contour, obj_kdtree, labels, centroids, k=175, nr_samples=None):
    """ Get the K nearest nuclei from the contour.
    
    Args:
        contour: input contour
        obj_kdtree (sklearn.neighbors.KDTree): KDTree of nuclei centroids
        labels: object labels (same index as the tree)
        centroids: objects coordinates (same index as the tree)
        k: return k nearest objects
    
    Returns:
        output_dst: distances of nearest objects
        output_labs: labels of nearest objects
        output_cents: coordinates of nearest objects
    """
    #! This needs optimisation. Consider geopandas.sindex.SpatialIndex.nearest
    if nr_samples < k:
        k = nr_samples

    contour = np.array(contour)
    dist, inds = obj_kdtree.query(contour, k=k)

    distances = {}
    if contour.shape[0] > 1:
        unique_inds = np.unique(inds).tolist()
        for ind in unique_inds:
            dist_subset = dist[inds == ind]
            min_dst = np.min(dist_subset)
            distances[ind] = min_dst
    else:
        for index in range(dist.shape[-1]):
            distances[inds[0, index]] = dist[0, index]

    distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}

    if len(list(distances.keys())) < k:
        k = len(list(distances.keys()))

    # output dict is {type: distance}
    output_labs = []
    output_cents = []
    output_dst = []
    dist_keys = list(distances.keys())
    dist_values = list(distances.values())
    for idx in range(k):
        grab_idx = dist_keys[idx]
        output_labs.append(labels[grab_idx])
        output_cents.append(centroids[grab_idx])
        output_dst.append(dist_values[idx])
    
    return output_dst, output_labs, output_cents


def get_nearest_within_radius(centroid, obj_kdtree, labels, r=50, nr_types=6):
    """ Get the objects within a fixed radius.
    
    Args:
        contour: input contour
        obj_kdtree (sklearn.neighbors.KDTree): KDTree of nuclei centroids
        labels: object labels (same index as the tree)
        r: return objects within fixed radius
    
    Returns:
        output_dict: frequencies of different nuclei types within radius of r
    """ 

    centroid = np.array(centroid)
    inds = obj_kdtree.query_radius(centroid, r=r, return_distance=False)
    inds = np.squeeze(inds).tolist()

    lab_list = []
    for ind in inds:
        lab_list.append(labels[ind])
    
    # output dict format: {type: frequency}
    output_dict = {}
    for idx in range(nr_types):
        output_dict[idx+1] = lab_list.count(idx+1)
    
    return output_dict


def get_kdtree(input_dict):
    """Convert input dictionary of results to KDTree.
    
    Args:
        input_dict: results dictionary.
    
    Returns: 
         centroid_kdtree (sklearn.neighbors.KDTree): KD-Tree of object centroids.
    
    """
    centroid_list = []
    label_list = []
    for key, values in input_dict.items():
        centroid_list.append(values["centroid"])
        label_list.append(values["type"])
    
    if len(centroid_list) > 0:
        centroid_array = np.array(centroid_list)
        centroid_kdtree = KDTree(centroid_array)
    else:
        centroid_kdtree = None
        label_list = None

    return centroid_kdtree, label_list, centroid_list


def inter_epi_dst(input_dict, obj_kdtree, labels, lab=2):
    """get the stats for distances between epithelial cells.
    
    Args:
        input_dict: 
        obj_kdtree (sklearn.neighbors.KDTree): KDTree of nuclei centroids in gland
        labels: labels of each nucleus
        lab: label to consider
    
    Returns:
        mean and std of inter-nuclei distances
    
    """
    centroid_list = []
    for values in input_dict.values():
        # find nearest object
        # only for epi
        if values["type"] == lab:
            centroid_list.append(values["centroid"])
    
    if len(centroid_list) > 1:
        dst_list = []
        for centroid in centroid_list:
            centroid = np.reshape(centroid, (1, 2))
            dst, _ = obj_kdtree.query(centroid, k=2)
            dst_list.append(dst[0,1])
    else:
        dst_list = [0]
    
    return dst_list


def get_dst_matrix(list_cnts, sorted=False):
    """Get the distance matrix. Measures the distance between all object contours.

    Args:
        list_cnts: input list of object contour coordinates.


    Returns:
        dst_matrix: NxN array of matrix of distances between objects.

    """
    nr_objs = len(list_cnts)
    dst_matrix = np.zeros([nr_objs, nr_objs])
    
    for i in range(nr_objs):
        cnt1 = list_cnts[i]
        for j in range(nr_objs):
            if i != j:
                cnt2 = list_cnts[j]
                dist = points_list_pairwise_edt(cnt1, cnt2)
                # distance between objects is the min dist between 2 contours
                dst_matrix[i, j] = np.min(dist)
    
    if sorted:
        dst_matrix = np.sort(dst_matrix, axis=-1)

    return dst_matrix


def get_bam(cnt, centroid):
    """Get the BAM (best alignment metric)."""
    # get enclosing ellipse
    ellipse_coords = get_enclosing_ellipse(cnt)
    # transform ellipse to circle
    circle_coords, alpha, a, b = ellipse_to_circle(ellipse_coords)
    # transform original object with same transformation
    trans_coords = apply_transform(cnt, centroid, alpha, a, b)
    # compute best alignment metric (BAM)
    bam_distance, _, _ = best_alignment_metric(
        circle_coords, trans_coords, show_plots=False
    )

    return bam_distance


def get_tissue_region_info_patch(cnt, mask, relax_pix=50, ds_factor=0.125):
    """Get the patch around a given contour by considering the relaxed bounding box."""

    # contour is at 0.5 mpp
    # mask is at 4.0 mpp
    # relax_pix operates at mask level

    # first convert contours to correct scale
    cnt = cnt * ds_factor
    cnt = cnt.astype('int')
    x,y,w,h = cv2.boundingRect(cnt)

    x = x - relax_pix
    y = y - relax_pix
    w = w + 2*relax_pix
    h = h + 2*relax_pix

    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x > (mask.shape[1] - relax_pix):
        x = mask.shape[1] - relax_pix
    if y > (mask.shape[0] - relax_pix):
        y = mask.shape[0] - relax_pix

    patch = mask[y: y+h, x: x+w]

    return patch


def get_patch_prop(region_info_patch, total_pix, labs):
    """Get the proportion of a certain tissue type in a labelled input patch."""
    output_list = []
    for lab in labs:
        lab_tmp = region_info_patch == lab
        nr_pix = np.sum(lab_tmp)
        
        if total_pix == 0:
            output_list.append(0)
        else:
            output_list.append(nr_pix / total_pix)
    
    return output_list