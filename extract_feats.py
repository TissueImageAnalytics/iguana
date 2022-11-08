"""extract_feats.py

Generates a graph along with node-level features from a WSI and it's instance-level mask

Usage:
  extract_feats.py [--k=<n>] [--start=<n>] [--end=<n>] [--wsi_ext=<str>]
  extract_feats.py (-h | --help)
  extract_feats.py --version

Options:
  -h --help           Show this string.
  --version           Show version.
  --k=<n>             Number of 'nearest' glands to consider. [default: 6]
  --start=<n>         Start position for file batching.
  --end=<n>           End position for file batching.
  --wsi_ext=<str>     Extension of the WSI file. [default: .svs]
"""

import logging 
import joblib
import pandas as pd
import numpy as np
import glob
import time
import os
import sys
import pickle
import shutil

from progress.bar import Bar as ProgressBar
from scipy.ndimage import measurements
from datetime import datetime
from docopt import docopt
from collections import defaultdict

from PIL import Image
Image.MAX_IMAGE_PIXELS = None # otherwise errors when reading large masks

from misc.utils import rm_n_mkdir
from misc.feat_utils import *

#! this needs to be factored out!
sys.path.append("/root/romesco_workspace/code/tiatoolbox")
from tiatoolbox.wsicore import wsireader

cv2.setNumThreads(0)  # otherwise multiprocessing hangs


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
    wsi_path, 
    tissue_type_path, 
    lamina_path, 
    output_path, 
    logging_path=None,
    cache_path=None,
    k=6,
    index_info=None,
):
    """
    Extract object-level features and construct graph

    Args:
        input_path: root path containing segmentation results for single file 
        mask_path: path to binary tissue masks
        wsi_path: path to original whole-slide images
        tissue_type_path: path to tissue type segmentation map
        lamina_path: path to lamina propria mask
        output_path: path where features w  ill be saved
        logging_path: path to where logging is saved
        cache_path: path where temporary files are saved
        k: number of nearest distances to consider (used in 'get_k_nearest_dst')
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

    # check whether file has not already been processed and laimina propria mask exists!
    if not os.path.exists(f"{save_path}/local_feats.csv") and os.path.exists(lamina_path + wsi_name + ".png") and wsi_name not in wsi_void_list:
        rm_n_mkdir(save_path)

        logger.info(f"Extracting features from {wsi_name} ...")
        logger.info(f"Batch index: {index_info} ...")

        # load the results data
        all_info = joblib.load(input_path)
        logger.info("Loaded data!")

        if "Gland" in all_info.keys():
            gland_info_ = all_info["Gland"]
        else:
            gland_info_ = {}
        if "Lumen" in all_info.keys():
            lumen_info_ = all_info["Lumen"]
        else:
            lumen_info_ = {}
        if "Nuclei" in all_info.keys():
            nuclei_info_ = all_info["Nuclei"]
        else:
            nuclei_info_ = {}
        proc_resolution = all_info["resolution"]
        del all_info

        # read the tissue mask
        tissue_mask = cv2.imread(mask_path + wsi_name + ".png", 0)
        tissue_mask_shape = tissue_mask.shape

        mask_ds_factor = 0.125
        gland_info = gland_info_
        lumen_info = lumen_info_
        nuclei_info = nuclei_info_
        
        tissue_type_mask = cv2.imread(tissue_type_path + wsi_name + ".png", 0)
        # read the segmentation maps
        if tissue_type_mask.shape != tissue_mask_shape:
            tissue_type_mask = cv2.resize(
                tissue_type_mask, 
                (tissue_mask.shape[1], tissue_mask.shape[0]), 
                interpolation=cv2.INTER_NEAREST
                )

        lamina_mask = cv2.imread(lamina_path + wsi_name + ".png", 0)
        if lamina_mask.shape != tissue_mask_shape:
            lamina_mask = cv2.resize(
                lamina_mask, 
                (tissue_mask.shape[1], tissue_mask.shape[0]), 
                interpolation=cv2.INTER_NEAREST
                )

        tissue_lab = measurements.label(tissue_mask)[0]
        unique_tissue = np.unique(tissue_lab).tolist()[1:]
        del tissue_mask

        # only consider non surface epithelium glands!
        gland_tissue_info = filter_coords_msk(gland_info, tissue_lab, scale=mask_ds_factor, label=1)
        surfepi_tissue_info = filter_coords_msk(gland_info, tissue_lab, scale=mask_ds_factor, label=2)
        lumen_tissue_info = filter_coords_msk(lumen_info, tissue_lab, scale=mask_ds_factor)
        nuclei_tissue_info, nuclei_lamina_info  = filter_coords_msk2(
            nuclei_info, 
            tissue_lab, 
            mask2 = lamina_mask,
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
        nuclei_lamina_cents = grab_centroids_type(nuclei_lamina_info)
        del nuclei_lamina_info

        # save to cache - reduces need for large RAM!
        cache_dir = f"{cache_path}/{wsi_name}"
        rm_n_mkdir(cache_dir)
        save_dict_regions(gland_tissue_info, cache_dir, 'gland')
        del gland_tissue_info
        save_dict_regions(surfepi_tissue_info, cache_dir, 'surfepi')
        del surfepi_tissue_info
        save_dict_regions(lumen_tissue_info, cache_dir, 'lumen')
        del lumen_tissue_info
        save_dict_regions(nuclei_tissue_info, cache_dir, 'nuclei_tissue')
        del nuclei_tissue_info
        save_dict_regions(nuclei_lamina_cents, cache_dir, 'nuclei_lp')
        del nuclei_lamina_cents
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
                with open(f"{cache_dir}/surfepi_{tissue_val}.pickle", 'rb') as f:
                    surfepi_info_subset = pickle.load(f)
                with open(f"{cache_dir}/lumen_{tissue_val}.pickle", 'rb') as f:
                    lumen_info_subset = pickle.load(f)
                with open(f"{cache_dir}/nuclei_tissue_{tissue_val}.pickle", 'rb') as f:
                    nuclei_tissue_info_subset = pickle.load(f)
                with open(f"{cache_dir}/nuclei_lp_{tissue_val}.pickle", 'rb') as f:
                    nuclei_lamina_info_subset = pickle.load(f)
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

                nr_nuclei_lamina = len(list(nuclei_lamina_info_subset.keys()))
                # convert centroids into kdtree for quick search
                nuclei_lamina_info_sub_kdtree, labels_nuc_lamina, centroids_nuc_lamina = get_kdtree(nuclei_lamina_info_subset)
                nuclei_tissue_info_sub_kdtree, labels_nuc_tissue, centroids_nuc_tissue = get_kdtree(nuclei_tissue_info_subset)
                gland_dst = get_dst_matrix(list(gland_cnts_subset.values()), sorted=True)
                gland_dst_subset = gland_dst[:, 1:nbors+1] # ignore first entry (diagnonal entry is always 0)

                gland_counter = 0
                for gland_idx, single_gland_info in gland_info_subset.items():
                    # initialise empty dictionary for features from single gland
                    local_feats_single = {}
                    gland_cnt = single_gland_info["contour"]
                    if gland_cnt.shape[0] > 4:
                        all_glands += 1
                        gland_contour_list.append(gland_cnts_subset[gland_idx])
                        tissue_idx_list.append(tissue_idx)
                        gland_morph_feats = get_contour_feats(gland_cnt)
                        gland_area = gland_morph_feats["Area"]
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
                         adipose_prop) = get_patch_prop(region_info_patch, total_pix, lab=[1,2,3,4,5,6,7,8])

                        # add features the dictionary
                        local_feats_single["Gland_Inflam_Prop"] = inflam_prop
                        local_feats_single["Gland_Mucous_Prop"] = mucous_prop
                        local_feats_single["Gland_Debris_Prop"] = debris_prop
                        local_feats_single["Gland_Normal_Prop"] = normal_prop
                        local_feats_single["Gland_Tumour_Prop"] = tumour_prop
                        local_feats_single["Gland_Adipose_Prop"] = adipose_prop
                        local_feats_single["Gland_Stroma_Prop"] = stroma_prop
                        local_feats_single["Gland_Muscle_Prop"] = muscle_prop

                        local_feats_single["Gland_BAM"] = gland_bam
                        for feat_name, value in gland_morph_feats.items():
                            local_feats_single[f"Gland-{feat_name}"] = value
                        for idx, value in enumerate(gland_distances):
                            local_feats_single[f"Gland-Dist{idx+1}"] = value

                        #! get the lumen info within the gland
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
                        for lumen_idx, single_lumen_info in lumen_within_gland_info.items():
                            lumen_cnt = single_lumen_info["contour"]
                            if lumen_cnt.shape[0] > 4:
                                nr_lumen += 1
                                lumen_morph_feats = get_contour_feats(lumen_cnt)
                                lumen_total_area += lumen_morph_feats["Area"]
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
                        local_feats_single["Lumen-Number"] = nr_lumen
                        local_feats_single["Lumen-Gland_Ratio"] = lumen_total_area / (gland_area + lumen_total_area)
                        local_feats_single["Lumen-BAM-Min"] = np.min(lumen_bam_list)
                        local_feats_single["Lumen-BAM-Max"] = np.max(lumen_bam_list)
                        local_feats_single["Lumen-BAM-Mean"] = np.mean(lumen_bam_list)
                        local_feats_single["Lumen-BAM-Std"] = np.std(lumen_bam_list)
                        ####
                        idx_count = 0
                        for feat_name in morph_feats_list:
                            local_feats_single[f"Lumen-{feat_name}-Min"] = np.min(lumen_morph_list[:, idx_count])
                            local_feats_single[f"Lumen-{feat_name}-Max"] = np.max(lumen_morph_list[:, idx_count])
                            local_feats_single[f"Lumen-{feat_name}-Mean"] = np.mean(lumen_morph_list[:, idx_count])
                            local_feats_single[f"Lumen-{feat_name}-Std"] = np.std(lumen_morph_list[:, idx_count])
                            idx_count += 1

                        #! get the nuclei info within the gland
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

                        for nuclei_idx, single_nuclei_info in nuclei_within_gland_info.items():
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

                        #! normalise counts by the gland area!
                        local_feats_single["Nuclei-Gland-EpiCount"] = count_epi / gland_area
                        local_feats_single["Nuclei-Gland-LymCount"] = count_lym / gland_area
                        local_feats_single["Nuclei-Gland-PlasCount"] = count_plas / gland_area
                        local_feats_single["Nuclei-Gland-NeutCount"] = count_neut / gland_area
                        local_feats_single["Nuclei-Gland-EosCount"] = count_eos / gland_area
                        local_feats_single["Nuclei-Gland-NucCount"] = count_nuclei / gland_area
                        ####
                        local_feats_single["Nuclei-InterEpi-Min"] = np.min(nuclei_inter_epi_dst)
                        local_feats_single["Nuclei-InterEpi-Max"] = np.max(nuclei_inter_epi_dst)
                        local_feats_single["Nuclei-InterEpi-Mean"] = np.mean(nuclei_inter_epi_dst)
                        local_feats_single["Nuclei-InterEpi-Std"] = np.std(nuclei_inter_epi_dst)
                        ####
                        local_feats_single["Nuclei-DistBoundary-Min"] = np.min(nuclei_dst_list)
                        local_feats_single["Nuclei-DistBoundary-Max"] = np.max(nuclei_dst_list)
                        local_feats_single["Nuclei-DistBoundary-Mean"] = np.mean(nuclei_dst_list)
                        local_feats_single["Nuclei-DistBoundary-Std"] = np.std(nuclei_dst_list)

                        local_feats_single["Nuclei-DistLumen-Min"] = np.min(nuclei_lumen_dst_list)
                        local_feats_single["Nuclei-DistLumen-Max"] = np.max(nuclei_lumen_dst_list)
                        local_feats_single["Nuclei-DistLumen-Mean"] = np.mean(nuclei_lumen_dst_list)
                        local_feats_single["Nuclei-DistLumen-Std"] = np.std(nuclei_lumen_dst_list)

                        idx_count = 0
                        for feat_name in morph_feats_list:
                            local_feats_single[f"Nuclei-{feat_name}-Min"] = np.min(nuclei_morph_list[:, idx_count])
                            local_feats_single[f"Nuclei-{feat_name}-Max"] = np.max(nuclei_morph_list[:, idx_count])
                            local_feats_single[f"Nuclei-{feat_name}-Mean"] = np.mean(nuclei_morph_list[:, idx_count])
                            local_feats_single[f"Nuclei-{feat_name}-Std"] = np.std(nuclei_morph_list[:, idx_count])
                            idx_count += 1

                        colocalisation_array = np.zeros([6, 6])
                        if nuclei_lamina_info_sub_kdtree is not None:
                            # now get the closest N nuclei to the gland and get the stats
                            # get the distances of all nuclei (within LP of tissue region) to the gland
                            nuclei_sample_nr = 300 # number of nuclei to select in lamina propria
                            # gland_nuc_dst_dict returns dict of form {type: distance}
                            gland_nuc_dst, nuc_labs, nuc_coords = get_k_nearest_from_contour(
                                gland_cnt, 
                                nuclei_lamina_info_sub_kdtree, 
                                labels_nuc_lamina, 
                                centroids_nuc_lamina,
                                k=nuclei_sample_nr,
                                nr_samples=nr_nuclei_lamina
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

                                # get co-localisation stats between given nucleus and all nuclei in tissue
                                colocalisation_radius = 200 # 100 micrometers
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
                            array_idx = ["Neut", "Epi", "Lym", "Plas", "Eos", "Conn"]
                            for coloc_idx1 in range(colocalisation_norm.shape[0]):
                                for coloc_idx2 in range(colocalisation_norm.shape[1]):
                                    coloc_value = colocalisation_norm[coloc_idx1, coloc_idx2]
                                    nuc_name1 = array_idx[coloc_idx1]
                                    nuc_name2 = array_idx[coloc_idx2]
                                    local_feats_single[f"Colocalisation-{nuc_name1}-{nuc_name2}"] = coloc_value

                            local_feats_single["Nuclei-LP-Lym-Prop"] = count_lym / len(gland_nuc_dst)
                            local_feats_single["Nuclei-LP-Plas-Prop"] = count_plas / len(gland_nuc_dst)
                            local_feats_single["Nuclei-LP-Neut-Prop"] = count_neut / len(gland_nuc_dst)
                            local_feats_single["Nuclei-LP-Eos-Prop"] = count_eos / len(gland_nuc_dst)
                            local_feats_single["Nuclei-LP-Conn-Prop"] = count_conn / len(gland_nuc_dst)
                            ##
                            local_feats_single["Nuclei-LP-Lym-Density"] = lym_mean_dst
                            local_feats_single["Nuclei-LP-Plas-Density"] = plas_mean_dst
                            local_feats_single["Nuclei-LP-Neut-Density"] = neut_mean_dst
                            local_feats_single["Nuclei-LP-Eos-Density"] = eos_mean_dst
                            local_feats_single["Nuclei-LP-Conn-Density"] = conn_mean_dst
                            local_feats_single["Nuclei-Inf-Density"] = inflam_mean_dst   

                        else:
                            local_feats_single["Nuclei-LP-Lym-Prop"] = 0
                            local_feats_single["Nuclei-LP-Plas-Prop"] = 0
                            local_feats_single["Nuclei-LP-Neut-Prop"] = 0
                            local_feats_single["Nuclei-LP-Eos-Prop"] = 0
                            local_feats_single["Nuclei-LP-Conn-Prop"] = 0
                            ##
                            local_feats_single["Nuclei-LP-Lym-Density"] = np.nan
                            local_feats_single["Nuclei-LP-Plas-Density"] = np.nan
                            local_feats_single["Nuclei-LP-Neut-Density"] = np.nan
                            local_feats_single["Nuclei-LP-Eos-Density"] = np.nan
                            local_feats_single["Nuclei-LP-Conn-Density"] = np.nan
                            local_feats_single["Nuclei-Inf-Density"] = np.nan

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

            # save local feats as dat file
            joblib.dump(local_feats, f"{save_path}/local_feats.dat")

            # save local feats to csv
            df_local_feats = pd.DataFrame(data=local_feats_agg)
            df_local_feats.to_csv(f"{save_path}/local_feats.csv", index=False)

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


def run(input_files, mask_path, wsi_path, tissue_type_path, lamina_path, output_path, logging_path, cache_path, k, idx_info, wsi_ext):
    """Extract features from all input files"""

    pbar = ProgressBar("Processing", max=len(input_files), width=48)

    #! this should be parallelised!
    for input_path in input_files:
        start = time.time()
        extract_features(
            input_path, 
            mask_path, 
            wsi_path, 
            tissue_type_path, 
            lamina_path, 
            output_path,
            logging_path,
            cache_path,
            k,
            idx_info,
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

    k = int(args["--k"])
    start_idx = int(args["--start"])
    end_idx = int(args["--end"])
    wsi_ext = args["--wsi_ext"]

    input_path = "/root/lsf_workspace/proc_slides/coad_tcga/cerberus/"
    mask_path = "/root/lsf_workspace/proc_slides/coad_tcga/masks/"
    wsi_path = ""
    tissue_type_path = "/root/lsf_workspace/proc_slides/coad_tcga/tissue/"
    lamina_path = "/root/lsf_workspace/proc_slides/coad_tcga/nongland/"
    output_path = "/root/lsf_workspace/proc_slides/coad_tcga/feats/"

    logging_path = "logging/"
    cache_path = "/root/dgx_workspace/cache/"

    input_files_ = glob.glob(input_path + "*.dat")

    input_files = []
    for input_file in input_files_:
        basename = os.path.basename(input_file)
        wsi_name = basename[:-4]
        save_path = output_path + wsi_name
        if not os.path.exists(f"{save_path}/local_feats.csv") and os.path.exists(lamina_path + wsi_name + ".png"):
            input_files.append(input_file)

    # rm_n_mkdir(output_path)
    input_files = input_files[start_idx:end_idx]

    # Run feature extraction
    run(input_files, mask_path, wsi_path, tissue_type_path, lamina_path, output_path, logging_path, cache_path, k, [start_idx, end_idx], wsi_ext)
# 