"""extract_feats.py

Generates a graph along with node-level features from a WSI and it's instance-level mask

Usage:
  extract_feats.py [--k=<n>] [--start=<n>] [--end=<n>] [--cerberus_dir=<path>] [--mask_dir=<path>] \
    [--tissue_type_dir=<path>] [--wsi_dir=<path>] [--output_dir=<path>] [--logging_dir=<path>] [--cache_path=<path>] [--focus_mode=<str>]
  extract_feats.py (-h | --help)
  extract_feats.py --version

Options:
  -h --help                 Show this string.
  --version                 Show version.
  --k=<n>                   Number of 'nearest' glands to consider. [default: 6]
  --start=<n>               Start position for file batching.
  --end=<n>                 End position for file batching.
  --cerberus_dir=<path>     Input directory to process containing results of Cerberus.
  --mask_dir=<path>         Directory where tissue mask are located.
  --tissue_type_dir=<path>  Directory where tissue type classification results from Cerberus are located.
  --wsi_dir=<path>          Directory where original WSIs are stored.
  --output_dir=<path>       Output directory where results where features will be saved.
  --logging_dir=<path>      Where python logging information will be output. [default: logging/]
  --cache_path=<path>       Cache location. [default: /root/dgx_workspace/cache/]
  --focus_mode=<str>        Region to consider for cell quantification. `lp` denotes lamina propria. [default: lp]
  
"""

import logging 
import joblib
import pandas as pd
import numpy as np
import glob
import time
import os
import cv2
import pickle
import shutil
import yaml

from progress.bar import Bar as ProgressBar
from scipy.ndimage import measurements
from datetime import datetime
from docopt import docopt
from collections import defaultdict

from PIL import Image
Image.MAX_IMAGE_PIXELS = None # otherwise errors when reading large masks

from misc.utils import rm_n_mkdir, get_focus_tissue
from misc.feat_utils import *

cv2.setNumThreads(0)  # otherwise multiprocessing hangs


morph_feats_list = [
    "area",
    "perimeter",
    "equiv-diameter",
    "extent",
    "convex-area",
    "solidity",
    "major-axis-length",
    "minor-axis-length",
    "eccentricity",
    "orientation",
    "ellipse-centre-x",
    "ellipse-centre-y",
    ]


def timeit(start, message):
    """Simple helper function for printing the time."""
    end = time.time()
    diff = end-start
    print(f"{message}: {round(diff)}")


def save_dict_regions(input_dict, cache_dir, name):
    """Save the dictionaries in each region to cache - reduces need for large RAM."""
    for tissue_idx, info in input_dict.items():
        with open(f"{cache_dir}/{name}_{tissue_idx}.pickle", 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)


def extract_features(
    input_path, 
    mask_path, 
    tissue_type_path, 
    wsi_path, 
    output_path, 
    logging_path=None,
    cache_path=None,
    k=6,
    index_info=None,
    focus_mode="lp"
):
    """
    Extract object-level features and construct graph

    Args:
        input_path: root path containing segmentation results for single file 
        mask_path: path to binary tissue masks
        tissue_type_path: path to tissue type segmentation map
        wsi_path: path to whole-slide images
        output_path: path where features w  ill be saved
        logging_path: path to where logging is saved
        cache_path: path where temporary files are saved
        k: number of nearest distances to consider (used in 'get_k_nearest_dst')
        index_info: information denoting the start and end indices of the batch- used for logging.
        focus_mode: whether to consider cell quantification in lamina propria (`lp`) or entire tissue.
            Lamina propria should be used when considering biopsies.
            
    """
    wsi_name = os.path.basename(input_path)
    wsi_name = wsi_name[:-4]

    wsi_void_list = []
    if wsi_name in wsi_void_list:
        print()
        logger.warning(f"Skipping {wsi_name} - in void list!")
        print(f"Skipping {wsi_name} - in void list!")

    # create output directory
    save_path = output_path + wsi_name
    dt_now = datetime.now()
    dt_string = dt_now.strftime("%d-%m-%Y_%H:%M:%S")
    wsi_logging_path = f"{logging_path}/{wsi_name}_{dt_string}_std.log"

    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=wsi_logging_path, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)

    # check whether file has not already been processed!
    if not os.path.exists(f"{save_path}/local_feats.csv") and wsi_name not in wsi_void_list:
        rm_n_mkdir(save_path)

        logger.info(f"Extracting features from {wsi_name} ...")
        logger.info(f"Batch index: {index_info} ...")

        # load the results data
        all_info = joblib.load(input_path)
        logger.info("Loaded data!")

        if "Gland" in all_info.keys():
            gland_info = all_info["Gland"]
        else:
            gland_info = {}
        if "Lumen" in all_info.keys():
            lumen_info = all_info["Lumen"]
        else:
            lumen_info = {}
        if "Nuclei" in all_info.keys():
            nuclei_info = all_info["Nuclei"]
        else:
            nuclei_info = {}
        # proc_resolution = all_info["proc_dimensions"]
        del all_info

        # read the tissue mask
        tissue_mask = cv2.imread(f"{mask_path}/{wsi_name}.png", 0)
        tissue_mask_shape = tissue_mask.shape

        # mask_ds_factor = tissue_mask_shape[0] / proc_resolution[0]
        mask_ds_factor = 0.125
        
        tissue_type_mask = cv2.imread(f"{tissue_type_path}/{wsi_name}.png", 0)
        # read the segmentation maps
        if tissue_type_mask.shape != tissue_mask_shape:
            tissue_type_mask = cv2.resize(
                tissue_type_mask, 
                (tissue_mask.shape[1], tissue_mask.shape[0]), 
                interpolation=cv2.INTER_NEAREST
                )

        wsi_filename = glob.glob(f"{wsi_path}/{wsi_name}*")[0]
        focus_mask = get_focus_tissue(wsi_filename, tissue_type_mask, gland_info, mode=focus_mode, ds_factor=8)
        logger.info("Generated focus mask.") # lamina propria for biopsy

        tissue_lab = measurements.label(tissue_mask)[0]
        unique_tissue = np.unique(tissue_lab).tolist()[1:]
        del tissue_mask

        # only consider non surface epithelium glands (gland label = 1)!
        gland_tissue_info = filter_coords_msk(gland_info, tissue_lab, scale=mask_ds_factor, label=1)
        lumen_tissue_info = filter_coords_msk(lumen_info, tissue_lab, scale=mask_ds_factor)
        # filter_coords_msk2 outputs 2 dicts: 
        # 1) nuclei info in each tissue region
        # 2) nuclei info in focus area of each tissue region (lamina propria for biopsy)
        nuclei_tissue_info, nuclei_focus_info  = filter_coords_msk2(
            nuclei_info, 
            tissue_lab, 
            mask2 = focus_mask,
            scale=mask_ds_factor,
            mode="centroid"
        )
        # free up memory
        del gland_info 
        del lumen_info 
        del nuclei_info
        logger.info("Filtered coordinates")

        # get the contour coords from glands and reduce the number of points for faster computation
        gland_tissue_cnts = grab_cnts(gland_tissue_info, ds_factor=4)
        # get the contour coords from glands and reduce the number of points for faster computation
        nuclei_focus_cents = grab_centroids_type(nuclei_focus_info)
        del nuclei_focus_info

        # save to cache - reduces need for large RAM!
        cache_dir = f"{cache_path}/{wsi_name}"
        rm_n_mkdir(cache_dir)
        save_dict_regions(gland_tissue_info, cache_dir, 'gland')
        del gland_tissue_info
        save_dict_regions(lumen_tissue_info, cache_dir, 'lumen')
        del lumen_tissue_info
        save_dict_regions(nuclei_tissue_info, cache_dir, 'nuclei_tissue')
        del nuclei_tissue_info
        save_dict_regions(nuclei_focus_cents, cache_dir, 'nuclei_focus')
        del nuclei_focus_cents
        save_dict_regions(gland_tissue_cnts, cache_dir, 'gland_cnts')
        del gland_tissue_cnts

        gland_contour_list = []
        tissue_idx_list = []
        local_feats = {}    
        # iterate over each tissue region and get local features
        all_glands = 0
        for tissue_idx, tissue_val in enumerate(unique_tissue):
            logger.info(
                "Extracting features from each tissue region (%d/%d)"
                % (tissue_idx, len(unique_tissue))
            )

            with open(f"{cache_dir}/gland_{tissue_val}.pickle", 'rb') as f:
                gland_info_subset = pickle.load(f)
            
            nr_glands = len(list(gland_info_subset.keys()))
            logger.info(f"Glands in tissue idx {tissue_idx}: {nr_glands}")
            # make sure there are at least 4 glands within the tissue region
            if nr_glands >= 4:
                with open(f"{cache_dir}/lumen_{tissue_val}.pickle", 'rb') as f:
                    lumen_info_subset = pickle.load(f)
                with open(f"{cache_dir}/nuclei_tissue_{tissue_val}.pickle", 'rb') as f:
                    nuclei_tissue_info_subset = pickle.load(f)
                with open(f"{cache_dir}/nuclei_focus_{tissue_val}.pickle", 'rb') as f:
                    nuclei_focus_info_subset = pickle.load(f)
                with open(f"{cache_dir}/gland_cnts_{tissue_val}.pickle", 'rb') as f:
                    gland_cnts_subset = pickle.load(f)

                # convert to df to help with fast retrieval
                nuclei_tissue_df = convert_to_df(nuclei_tissue_info_subset)
                lumen_tissue_df = convert_to_df(lumen_info_subset, return_type=False)
                del lumen_info_subset

                # get the distances to the k nearest glands
                nr_neighbours = k
                if nr_glands <= nr_neighbours:
                    nbors = nr_glands - 1
                else:
                    nbors = nr_neighbours

                nr_nuclei_focus = len(list(nuclei_focus_info_subset.keys()))
                # convert centroids into kdtree for quick search
                nuclei_focus_info_sub_kdtree, labels_nuc_focus, centroids_nuc_focus = get_kdtree(nuclei_focus_info_subset)
                nuclei_tissue_info_sub_kdtree, labels_nuc_tissue, centroids_nuc_tissue = get_kdtree(nuclei_tissue_info_subset)
                gland_dst = get_dst_matrix(list(gland_cnts_subset.values()), sorted=True)
                gland_dst_subset = gland_dst[:, 1:nbors+1] # ignore first entry (diagnonal entry is always 0)

                gland_counter = 0
                for gland_idx, single_gland_info in gland_info_subset.items():
                    # initialise empty dictionary for features from single gland
                    local_feats_single = {}
                    local_feats_single["obj_id"] = gland_idx # unqiue object identifier
                    gland_cnt = single_gland_info["contour"]
                    if gland_cnt.shape[0] > 4:
                        all_glands += 1
                        gland_contour_list.append(gland_cnts_subset[gland_idx])
                        tissue_idx_list.append(tissue_idx)
                        gland_morph_feats = get_contour_feats(gland_cnt)
                        gland_area = gland_morph_feats["area"]
                        gland_centroid = single_gland_info["centroid"]
                        gland_bbox = cv2.boundingRect(gland_cnt)

                        gland_bam = get_bam(gland_cnt, gland_centroid)
                        gland_distances = gland_dst_subset[gland_counter, :].tolist()

                        if len(gland_distances) < nr_neighbours:
                            for dst_idx in range(nr_neighbours - nbors):
                                gland_distances.append(np.nan)

                        # get local tissue info 
                        region_info_patch = get_tissue_region_info_patch(gland_cnt, tissue_type_mask) # get tissue patch
                        binary_patch = region_info_patch.copy()
                        binary_patch[binary_patch>0] = 1
                        total_pix = np.sum(binary_patch)
                        # get proportions of each tissue type around the cropped gland
                        (tumour_prop, 
                         normal_prop, 
                         inflam_prop, 
                         muscle_prop, 
                         stroma_prop, 
                         debris_prop, 
                         mucous_prop, 
                         adipose_prop) = get_patch_prop(region_info_patch, total_pix, labs=[1,2,3,4,5,6,7,8])

                        # add features the dictionary
                        local_feats_single["gland_inflam_prop"] = inflam_prop
                        local_feats_single["gland_mucous_prop"] = mucous_prop
                        local_feats_single["gland_debris_prop"] = debris_prop
                        local_feats_single["gland_normal_prop"] = normal_prop
                        local_feats_single["gland_tumour_prop"] = tumour_prop
                        local_feats_single["gland_adipose_prop"] = adipose_prop
                        local_feats_single["gland_stroma_prop"] = stroma_prop
                        local_feats_single["gland_muscle_prop"] = muscle_prop

                        local_feats_single["gland_bam"] = gland_bam
                        for feat_name, value in gland_morph_feats.items():
                            local_feats_single[f"gland-{feat_name}"] = value
                        for idx, value in enumerate(gland_distances):
                            local_feats_single[f"gland-dist{idx+1}"] = value

                        #* get the lumen info within the gland
                        filtered_lumen_tissue = lumen_tissue_df.loc[
                            (lumen_tissue_df['cx'] >= gland_bbox[0]) & 
                            (lumen_tissue_df['cy'] >= gland_bbox[1]) &
                            (lumen_tissue_df['cx'] <= (gland_bbox[0] + gland_bbox[2])) & 
                            (lumen_tissue_df['cy'] <= (gland_bbox[1] + gland_bbox[3]))
                            ]
                        lumen_within_gland_info = filter_coords_cnt(
                            filtered_lumen_tissue, gland_cnt, return_type=False
                        )

                        lumen_bam_list = []
                        lumen_morph_list = []
                        nr_lumen = 0
                        lumen_total_area = 0
                        for _, single_lumen_info in lumen_within_gland_info.items():
                            lumen_cnt = single_lumen_info["contour"]
                            if lumen_cnt.shape[0] > 4:
                                nr_lumen += 1
                                lumen_morph_feats = get_contour_feats(lumen_cnt)
                                lumen_total_area += lumen_morph_feats["area"]
                                lumen_centroid = single_lumen_info["centroid"]
                                lumen_bam = get_bam(lumen_cnt, lumen_centroid)

                                lumen_bam_list.append(lumen_bam)
                                lumen_morph_list_tmp = []
                                for feat_name, value in lumen_morph_feats.items():
                                    lumen_morph_list_tmp.append(value)
                                lumen_morph_list.append(lumen_morph_list_tmp)

                        if nr_lumen == 0:
                            lumen_bam_list = np.zeros([1])
                            lumen_bam_list[:] = np.nan
                            lumen_morph_list = np.zeros([1, 12])
                            lumen_morph_list[:] = np.nan
                        else:
                            lumen_bam_list = np.array(lumen_bam_list)
                            lumen_morph_list = np.array(lumen_morph_list)
                        ####
                        local_feats_single["lumen-number"] = nr_lumen
                        local_feats_single["lumen-gland_ratio"] = lumen_total_area / (gland_area + lumen_total_area)
                        local_feats_single["lumen-bam-min"] = np.min(lumen_bam_list)
                        local_feats_single["lumen-bam-max"] = np.max(lumen_bam_list)
                        local_feats_single["lumen-bam-mean"] = np.mean(lumen_bam_list)
                        local_feats_single["lumen-bam-std"] = np.std(lumen_bam_list)
                        ####
                        idx_count = 0
                        for feat_name in morph_feats_list:
                            local_feats_single[f"lumen-{feat_name}-min"] = np.min(lumen_morph_list[:, idx_count])
                            local_feats_single[f"lumen-{feat_name}-max"] = np.max(lumen_morph_list[:, idx_count])
                            local_feats_single[f"lumen-{feat_name}-mean"] = np.mean(lumen_morph_list[:, idx_count])
                            local_feats_single[f"lumen-{feat_name}-mtd"] = np.std(lumen_morph_list[:, idx_count])
                            idx_count += 1

                        #* get the nuclei info within the gland
                        filtered_nuclei_tissue = nuclei_tissue_df.loc[
                            (nuclei_tissue_df['cx'] >= gland_bbox[0]) & 
                            (nuclei_tissue_df['cy'] >= gland_bbox[1]) &
                            (nuclei_tissue_df['cx'] <= (gland_bbox[0] + gland_bbox[2])) & 
                            (nuclei_tissue_df['cy'] <= (gland_bbox[1] + gland_bbox[3]))
                            ]
                        nuclei_within_gland_info = filter_coords_cnt(
                            filtered_nuclei_tissue, gland_cnt, mode="centroid",
                        )

                        nuclei_within_gland_kdtree, nuclei_within_gland_labs, _ = get_kdtree(nuclei_within_gland_info)
                        nuclei_inter_epi_dst = inter_epi_dst(
                            nuclei_within_gland_info, 
                            nuclei_within_gland_kdtree, 
                            nuclei_within_gland_labs
                            )

                        nuclei_morph_list = []
                        nuclei_dst_list = []
                        nuclei_lumen_dst_list = []
                        count_epi = 0
                        count_neut = 0
                        count_lym = 0
                        count_plas = 0
                        count_eos = 0
                        count_nuclei = 0

                        for single_nuclei_info in nuclei_within_gland_info.values():
                            #### Get the counts of different nuclei within each gland
                            nuclei_type = single_nuclei_info["type"]
                            ####
                            nuclei_cnt = single_nuclei_info["contour"]
                            if nuclei_cnt.shape[0] > 4:
                                if nuclei_type == 1:
                                    count_neut += 1
                                elif nuclei_type == 2:
                                    count_epi += 1
                                elif nuclei_type == 3:
                                    count_lym += 1
                                elif nuclei_type == 4:
                                    count_plas += 1
                                elif nuclei_type == 5:
                                    count_eos += 1
                                count_nuclei += 1

                                # don't compute BAM here as it takes too long!
                                nuclei_morph_feats = get_contour_feats(nuclei_cnt)
                                nuclei_centroid = single_nuclei_info["centroid"]
                                boundary_dst = get_boundary_distance(gland_cnt, nuclei_centroid)
                                lumen_dst = get_lumen_distance(lumen_within_gland_info, nuclei_centroid)
                                nuclei_dst_list.append(boundary_dst)
                                nuclei_lumen_dst_list.append(lumen_dst)

                                nuclei_morph_list_tmp = []
                                for feat_name, value in nuclei_morph_feats.items():
                                    nuclei_morph_list_tmp.append(value)
                                nuclei_morph_list.append(nuclei_morph_list_tmp)

                        # convert to numpy array
                        if count_nuclei == 0:
                            nuclei_dst_list = np.zeros([1])
                            nuclei_lumen_dst_list = np.zeros([1])
                            nuclei_morph_list = np.zeros([1, 12])
                        else:
                            nuclei_dst_list = np.array(nuclei_dst_list)
                            nuclei_lumen_dst_list = np.array(nuclei_lumen_dst_list)
                            nuclei_morph_list = np.array(nuclei_morph_list)

                        #* normalise counts by the gland area!
                        local_feats_single["nuclei-gland-epi-count"] = count_epi / gland_area
                        local_feats_single["nuclei-gland-lym-count"] = count_lym / gland_area
                        local_feats_single["nuclei-gland-plas-count"] = count_plas / gland_area
                        local_feats_single["nuclei-gland-neut-count"] = count_neut / gland_area
                        local_feats_single["nuclei-gland-eos-count"] = count_eos / gland_area
                        local_feats_single["nuclei-gland-nuc-count"] = count_nuclei / gland_area
                        ###
                        local_feats_single["nuclei-inter-epi-min"] = np.min(nuclei_inter_epi_dst)
                        local_feats_single["nuclei-inter-epi-max"] = np.max(nuclei_inter_epi_dst)
                        local_feats_single["nuclei-inter-epi-mean"] = np.mean(nuclei_inter_epi_dst)
                        local_feats_single["nuclei-inter-epi-std"] = np.std(nuclei_inter_epi_dst)
                        ###
                        local_feats_single["nuclei-dist-boundary-min"] = np.min(nuclei_dst_list)
                        local_feats_single["nuclei-dist-boundary-max"] = np.max(nuclei_dst_list)
                        local_feats_single["nuclei-dist-boundary-mean"] = np.mean(nuclei_dst_list)
                        local_feats_single["nuclei-dist-boundary-std"] = np.std(nuclei_dst_list)
                        ###
                        local_feats_single["nuclei-dist-lumen-min"] = np.min(nuclei_lumen_dst_list)
                        local_feats_single["nuclei-dist-lumen-max"] = np.max(nuclei_lumen_dst_list)
                        local_feats_single["nuclei-dist-lumen-mean"] = np.mean(nuclei_lumen_dst_list)
                        local_feats_single["nuclei-dist-lumen-std"] = np.std(nuclei_lumen_dst_list)
                        idx_count = 0
                        for feat_name in morph_feats_list:
                            local_feats_single[f"nuclei-{feat_name}-min"] = np.min(nuclei_morph_list[:, idx_count])
                            local_feats_single[f"nuclei-{feat_name}-max"] = np.max(nuclei_morph_list[:, idx_count])
                            local_feats_single[f"nuclei-{feat_name}-mean"] = np.mean(nuclei_morph_list[:, idx_count])
                            local_feats_single[f"nuclei-{feat_name}-std"] = np.std(nuclei_morph_list[:, idx_count])
                            idx_count += 1

                        colocalisation_array = np.zeros([6, 6])
                        if nuclei_focus_info_sub_kdtree is not None:
                            # now get the closest N nuclei to the gland and get the stats
                            # get the distances of all nuclei (within focus region of tissue) to the gland
                            nuclei_sample_nr = 300 # number of nuclei to select in focus region (lamina propria for biopsy)
                            # gland_nuc_dst_dict returns dict of form {type: distance}
                            gland_nuc_dst, nuc_labs, nuc_coords = get_k_nearest_from_contour(
                                gland_cnt, 
                                nuclei_focus_info_sub_kdtree, 
                                labels_nuc_focus, 
                                centroids_nuc_focus,
                                k=nuclei_sample_nr,
                                nr_samples=nr_nuclei_focus
                                )

                            count_neut = 0
                            count_lym = 0
                            count_plas = 0
                            count_eos = 0
                            count_conn = 0
                            neut_dst_list = []
                            lym_dst_list = []
                            plas_dst_list = []
                            eos_dst_list = []
                            conn_dst_list = []
                            inflam_dst_list = []
                            for nuc_idx, nuc_dst in enumerate(gland_nuc_dst):
                                nuc_type = nuc_labs[nuc_idx]
                                nuc_centroid = nuc_coords[nuc_idx]
                                if nuc_type == 1:
                                    count_neut += 1
                                    neut_dst_list.append(nuc_dst)
                                    inflam_dst_list.append(nuc_dst)
                                elif nuc_type == 3:
                                    count_lym += 1
                                    lym_dst_list.append(nuc_dst)
                                    inflam_dst_list.append(nuc_dst)
                                elif nuc_type == 4:
                                    count_plas += 1
                                    plas_dst_list.append(nuc_dst)
                                    inflam_dst_list.append(nuc_dst)
                                elif nuc_type == 5:
                                    count_eos += 1
                                    eos_dst_list.append(nuc_dst)
                                    inflam_dst_list.append(nuc_dst)
                                elif nuc_type == 6:
                                    count_conn += 1
                                    conn_dst_list.append(nuc_dst)

                                # get co-localisation stats between given nucleus and all other nuclei in tissue
                                colocalisation_radius = 200 # 100 micrometers (working at 0.5 microns per pixel)
                                tissue_nuc_freq = get_nearest_within_radius(
                                    np.expand_dims(nuc_centroid, 0),
                                    nuclei_tissue_info_sub_kdtree, 
                                    labels_nuc_tissue, 
                                    r=colocalisation_radius,
                                    nr_types=6
                                    )
                                
                                for nuc_type2, nuc_freq in tissue_nuc_freq.items():
                                    colocalisation_array[nuc_type-1][nuc_type2-1] += nuc_freq
                            
                            # gives a good idea for how densely packed the inflammatory nuclei are
                            if len(inflam_dst_list) != 0:
                                inflam_mean_dst = sum(inflam_dst_list) / len(inflam_dst_list)
                            else:
                                inflam_mean_dst = np.nan
                            ##
                            if len(neut_dst_list) != 0:
                                neut_mean_dst = sum(neut_dst_list) / len(neut_dst_list)
                            else:
                                neut_mean_dst = np.nan
                            ##
                            if len(lym_dst_list) != 0:
                                lym_mean_dst = sum(lym_dst_list) / len(lym_dst_list)
                            else:
                                lym_mean_dst = np.nan
                            ##
                            if len(plas_dst_list) != 0:
                                plas_mean_dst = sum(plas_dst_list) / len(plas_dst_list)
                            else:
                                plas_mean_dst = np.nan
                            ##
                            if len(eos_dst_list) != 0:
                                eos_mean_dst = sum(eos_dst_list) / len(eos_dst_list)
                            else:
                                eos_mean_dst = np.nan
                            ##
                            if len(conn_dst_list) != 0:
                                conn_mean_dst = sum(conn_dst_list) / len(conn_dst_list)
                            else:
                                conn_mean_dst = np.nan

                            colocalisation_norm = []
                            for row_idx in range(colocalisation_array.shape[0]):
                                row_tmp = colocalisation_array[row_idx, :]
                                if np.sum(row_tmp) == 0:
                                    colocalisation_norm.append(row_tmp)
                                else:
                                    colocalisation_norm.append(row_tmp / np.sum(row_tmp))
                            colocalisation_norm = np.array(colocalisation_norm)
                            nuc_classes = ["neut", "epi", "lym", "plas", "eos", "conn"]
                            for coloc_idx1 in range(colocalisation_norm.shape[0]):
                                for coloc_idx2 in range(colocalisation_norm.shape[1]):
                                    coloc_value = colocalisation_norm[coloc_idx1, coloc_idx2]
                                    nuc_name1 = nuc_classes[coloc_idx1]
                                    nuc_name2 = nuc_classes[coloc_idx2]
                                    local_feats_single[f"colocalisation-{nuc_name1}-{nuc_name2}"] = coloc_value

                            local_feats_single["nuclei-focus-lym-prop"] = count_lym / len(gland_nuc_dst)
                            local_feats_single["nuclei-focus-plas-prop"] = count_plas / len(gland_nuc_dst)
                            local_feats_single["nuclei-focus-neut-prop"] = count_neut / len(gland_nuc_dst)
                            local_feats_single["nuclei-focus-eos-prop"] = count_eos / len(gland_nuc_dst)
                            local_feats_single["nuclei-focus-conn-prop"] = count_conn / len(gland_nuc_dst)
                            ###
                            local_feats_single["nuclei-focus-lym-density"] = lym_mean_dst
                            local_feats_single["nuclei-focus-plas-density"] = plas_mean_dst
                            local_feats_single["nuclei-focus-neut-density"] = neut_mean_dst
                            local_feats_single["nuclei-focus-eos-density"] = eos_mean_dst
                            local_feats_single["nuclei-focus-conn-density"] = conn_mean_dst
                            local_feats_single["nuclei-inf-density"] = inflam_mean_dst   

                        else:
                            nuc_classes = ["neut", "epi", "lym", "plas", "eos", "conn"]
                            for nuc_name1 in nuc_classes:
                                for nuc_name2 in nuc_classes:
                                    local_feats_single[f"colocalisation-{nuc_name1}-{nuc_name2}"] = 0
                            ###    
                            local_feats_single["nuclei-focus-lym-prop"] = 0
                            local_feats_single["nuclei-focus-plas-prop"] = 0
                            local_feats_single["nuclei-focus-neut-prop"] = 0
                            local_feats_single["nuclei-focus-eos-prop"] = 0
                            local_feats_single["nuclei-focus-conn-prop"] = 0
                            ###
                            local_feats_single["nuclei-focus-lym-density"] = np.nan
                            local_feats_single["nuclei-focus-plas-density"] = np.nan
                            local_feats_single["nuclei-focus-neut-density"] = np.nan
                            local_feats_single["nuclei-focus-eos-density"] = np.nan
                            local_feats_single["nuclei-focus-conn-density"] = np.nan
                            local_feats_single["nuclei-inf-density"] = np.nan

                        local_feats[gland_idx] = local_feats_single

                    gland_counter += 1

        # at least 4 glands in entire tissue!
        if all_glands > 4:
            # aggregate local feats to make it easy to work with
            local_feats_agg = defaultdict(list)
            for inst_info in local_feats.values():
                for feat_name, feat_value in inst_info.items():
                    local_feats_agg[feat_name].append(feat_value)

            # get the distance matrix
            gland_dst_array = get_dst_matrix(gland_contour_list)
            np.save(f"{save_path}/dst_matrix.npy", gland_dst_array)

            # get tissue idx array
            tissue_idx_array = np.array(tissue_idx_list)
            np.save(f"{save_path}/tissue_idx.npy", tissue_idx_array)
            
            #* format of features
            # {
            #   obj_id: [id1,    id2,    ...],    
            #   feat1:  [value1, value2, ...],
            #   feat2:  [value1, value2, ...],
            #   feat3:  [value1, value2, ...]
            #}
            joblib.dump(local_feats_agg, f"{save_path}/local_feats.dat") # save local feats as dat file
            df_local_feats = pd.DataFrame(data=local_feats_agg)
            df_local_feats.to_csv(f"{save_path}/local_feats.csv", index=False) # save local feats to csv
            
            # dump the complete list of feature names - this will overwrite each time, but is a low cost task
            feat_names = list(local_feats_agg.keys())
            with open("features_all.yml", "w") as fptr:
                yaml.dump({"features": feat_names}, fptr, default_flow_style=False)

            # below is a sanity check!
            dst_matrix_shape = gland_dst_array.shape
            len_tissue_idx = len(tissue_idx_list)
            logger.info(f'Number of glands: {all_glands}')
            logger.info(f'Length tissue index: {len_tissue_idx}')
            logger.info(f'Shape dst matrix: {dst_matrix_shape}')

            logger.info('Finished!')

            logger.handlers.clear() # close logger
            shutil.rmtree(cache_dir, ignore_errors=True)
        else:
            logger.warning('Less than 4 glands in the image - not saving features!')
            logger.handlers.clear() # close logger
            # remove directory if < 4 glands exist!
            shutil.rmtree(save_path, ignore_errors=True)
            shutil.rmtree(cache_dir, ignore_errors=True)
    else:
        logger.warning(f"Already processed {wsi_name} - SKIP!")
        logger.handlers.clear() # close logger


def run(input_files, mask_path, tissue_type_path, wsi_path, output_path, logging_path, cache_path, k, idx_info, focus_mode):
    """Extract features from all input files"""

    pbar = ProgressBar("Processing", max=len(input_files), width=48)

    #! this should be parallelised!
    for input_path in input_files:
        start = time.time()
        extract_features(
            input_path, 
            mask_path, 
            tissue_type_path, 
            wsi_path, 
            output_path,
            logging_path,
            cache_path,
            k,
            idx_info,
            focus_mode,
        )
        end = time.time()
        print('Processing Time:', end-start)
        pbar.next()
        print()
        print('='*80)
        print('='*80)
    pbar.finish()


# ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = docopt(__doc__)

    # start and end idx are used for subsetting the input file list for processing
    start_idx = int(args["--start"]) 
    end_idx = int(args["--end"])
    k = int(args["--k"])
    focus_mode = args["--focus_mode"] # use `lp` (lamina propria) for biopsy
    
    # directory / path info
    cerberus_path = args["--cerberus_dir"]
    mask_path = args["--mask_dir"]
    tissue_type_path = args["--tissue_type_dir"]
    wsi_path = args["--wsi_dir"]
    output_path = args["--output_dir"]   
    logging_path = args["--logging_dir"]
    cache_path = args["--cache_path"] 

    input_files_ = glob.glob(cerberus_path + "/*.dat") # cerberus segmentation results

    input_files = []
    for input_file in input_files_:
        basename = os.path.basename(input_file)
        wsi_name = basename[:-4]
        save_path = output_path + wsi_name
        if not os.path.exists(f"{save_path}/local_feats.csv"):
            input_files.append(input_file)

    # create output directory
    if not os.path.exists(output_path):
        rm_n_mkdir(output_path)
    
    # create logging directory
    if not os.path.exists(logging_path):
        rm_n_mkdir(logging_path)
        
    input_files = input_files[start_idx:end_idx]

    # run feature extraction
    run(input_files, mask_path, tissue_type_path, wsi_path, output_path, logging_path, cache_path, k, [start_idx, end_idx], focus_mode)