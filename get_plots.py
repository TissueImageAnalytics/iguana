"""get_plots.py

Generate plots of results.

Usage:
  extract_feats.py [--dataset=<str>] [--output_dir=<path>]
  extract_feats.py (-h | --help)
  extract_feats.py --version

Options:
  -h --help             Show this string.
  --version             Show version.
  --dataset=<str>       Which dataset to generate the plots for.
  --output_dir=<path>   Path to where the plots will be saved. [default: output/plots]
  
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from tqdm import tqdm

from docopt import docopt

from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, precision_score, recall_score

from metrics.stats_utils import get_sens_spec_metrics
from misc.utils import rm_n_mkdir


def get_labels_scores(wsi_names, scores, gt, binarize=True):
    """Get the label info according to input list of WSI names."""
    gt = pd.read_csv(gt)
    labels_output = []
    scores_output = []
    nr_normal = 0
    for idx, wsi_name in enumerate(wsi_names):
        score = scores[idx]
        gt_subset = gt[gt["wsi_id"] == wsi_name]
        lab = list(gt_subset["label_id"])
        if len(lab) > 0:
            lab = int(lab[0])
            if binarize:
                if lab > 0:
                    lab = 1
            # count the number of samples
            if lab == 0:
                nr_normal += 1
            labels_output.append(lab)
            scores_output.append(score)
    normal_prop = nr_normal / len(labels_output)
    return labels_output, scores_output, normal_prop
    
    
def screening_info(y, z, npts = None):
    """Get information regarding clinical utility. This is utlilised to 
    create a plot indicating the percentange of slides that require review
    to obtain a certain performance.
    
    """
    
    if npts is None:
        Z = np.sort(list(set(z)))
    else:
        Z = np.linspace(np.min(z), np.max(z), npts)
    eps = 1e-6
    Z = np.append(np.append(Z[0] - eps, Z), Z[-1] + eps)
    assert set(y) == set((0,1)) #labels must be 0,1
    M = []
    for t in tqdm(Z):
        yp = (z > t)
        tn, fp, fn, tp = confusion_matrix(y, yp).ravel()
        npv = precision_score(y, yp,pos_label = 0, zero_division = 1)
        rfo = 1 - npv # rate of false omission
        se = recall_score(y, yp)
        reviewed = np.mean(yp)
        screened_out = (fn+tn) / len(y)    
        m = (screened_out, rfo, se, reviewed)
        M.append(m)
    M = np.array(M)

    return M


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Args:
        ax (matplotlib.pyplot.axis): The axis we want to draw our plot on.

    data (dict): A dictionary containing the data we want to plot. 
        Keys are the names of the data, the items is a list of the values.
        
        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors (array) : A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width (float) : The width of a bar group.

    single_width (float): The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend (book): If this is set to true, a legend will be added to the axis.
    
    """

    # check if colors where provided, otherwise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # number of bars per group
    n_bars = len(data)

    # the width of a single bar
    bar_width = total_width / n_bars

    # list containing handles for the drawn bars, used for the legend
    bars = []

    # iterate over all data
    for i, (name, values_) in enumerate(data.items()):
        values = values_[0]
        std_values = values_[1]
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)], yerr=std_values[x])

        # add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])

    # draw legend if we need
    if legend:
        ax.legend(bars, data.keys())
        

def get_vis_roc(y, y_pred, mean_fp):
    """Get the ROC curve for visualisation."""
    fp, tp, _ = roc_curve(y, y_pred)
    roc_auc = auc(fp, tp)

    interp_tp = np.interp(mean_fp, fp, tp)
    interp_tp[0] = 0.0

    return fp, interp_tp, roc_auc


def get_vis_pr(y, y_pred, mean_fp):
    """Get the PR curve for visualisation."""
    pr, re, _ = precision_recall_curve(y, y_pred)
    avg_pr = auc(re, pr)

    interp_pr = np.interp(mean_fp, pr, re)
    interp_pr[0] = 1.0

    return fp, interp_pr, avg_pr


def plot_curve(tp_list, score_list, mean_fp, ax, model_name, color, alpha, lims, show_legend, axes_show, ticks_visible, mode="roc"):
    """Plot the ROC or PR curve. mode must be either `roc` or `pr`."""

    mode = mode.lower()
    if mode not in ["roc", "pr"]:
        raise ValueError("mode must be either `roc` or `pr`!")
    
    mean_tp = np.mean(tp_list, axis=0)
    if mode == "roc":
        mean_tp[-1] = 1.0
    else:
        mean_tp[-1] = 0.0
    mean_score = np.mean(score_list)
    std_score = np.std(score_list)
    
    label_desc_dict = {"roc": "AUC-ROC", "pr": "AUC-PR"} # used for the legend in the plot
    label_desc = label_desc_dict[mode] 
    ax.plot(
        mean_fp,
        mean_tp,
        color=color,
        label=f"{model_name}: {label_desc} = %0.4f $\pm$ %0.4f" % (mean_score, std_score),
        lw=2,
        alpha=0.8,
    )

    std_tp = np.std(tp_list, axis=0)
    tp_upper = np.minimum(mean_tp + std_tp, 1)
    tp_lower = np.maximum(mean_tp - std_tp, 0)
    ax.fill_between(
        mean_fp,
        tp_lower,
        tp_upper,
        color=color,
        alpha=alpha,
    )

    ax.set(
        xlim=lims[0],
        ylim=lims[1],
    )
    if show_legend:
        if mode == "roc":
            ax.legend(loc="lower right")
        else:
            ax.legend(loc="lower left")
    
    ax.spines['right'].set_visible(axes_show[0])
    ax.spines['top'].set_visible(axes_show[1])
    ax.spines['left'].set_visible(axes_show[2])
    ax.spines['bottom'].set_visible(axes_show[3])

    if ticks_visible == False:
        ax.set_xticks([])
        ax.set_yticks([])
        for _,s in ax.spines.items():
            s.set_linewidth(4)
            s.set_color('black')
    else:
        if mode == "roc":
            ax.add_patch(Rectangle((0, 0.7), 0.3, 0.3, fill=False, edgecolor='black', lw=1, linestyle='--'))
        else:
            ax.add_patch(Rectangle((0.7, 0.7), 0.3, 0.3, fill=False, edgecolor='black', lw=1, linestyle='--'))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if mode == "roc":
            plt.xlabel('1-Specificity', fontsize=14)
            plt.ylabel('Sensitivity', fontsize=14)
        else:
            plt.xlabel('Recall', fontsize=14)
            plt.ylabel('Precision', fontsize=14)
        for _,s in ax.spines.items():
            s.set_linewidth(2)
            s.set_color('black')

    return ax

# ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = docopt(__doc__)

    dataset_name = args["--dataset"]
    save_dir = args["--output_dir"]
    
    if not os.path.exists(save_dir):
        rm_n_mkdir(save_dir)
    
    results_iguana = {
        1: f"/mnt/gpfs01/lsf-workspace/tialab-simon/output/gland_graphs/model_output/pna/{dataset_name}/fold1/v1.5-nospiro/results.csv",
        2: f"/mnt/gpfs01/lsf-workspace/tialab-simon/output/gland_graphs/model_output/pna/{dataset_name}/fold2/v1.5-nospiro/results.csv",
        3: f"/mnt/gpfs01/lsf-workspace/tialab-simon/output/gland_graphs/model_output/pna/{dataset_name}/fold3/v1.5-nospiro/results.csv",
        } 
    results_rf = {
        1: f"output_hc/RF/{dataset_name}/results_1.csv",
        2: f"output_hc/RF/{dataset_name}/results_2.csv",
        3: f"output_hc/RF/{dataset_name}/results_3.csv",
        } 
    results_idars_avg = {
        1: f"/mnt/gpfs01/lsf-workspace/tialab-simon/output/gland_graphs/model_output/idars/avg/{dataset_name}/fold1/results.csv",
        2: f"/mnt/gpfs01/lsf-workspace/tialab-simon/output/gland_graphs/model_output/idars/avg/{dataset_name}/fold2/results.csv",
        3: f"/mnt/gpfs01/lsf-workspace/tialab-simon/output/gland_graphs/model_output/idars/avg/{dataset_name}/fold3/results.csv",
        } 
    results_clam = {
        1: f"/mnt/gpfs01/lsf-workspace/tialab-simon/output/gland_graphs/model_output/clam/{dataset_name}/fold1/results.csv",
        2: f"/mnt/gpfs01/lsf-workspace/tialab-simon/output/gland_graphs/model_output/clam/{dataset_name}/fold2/results.csv",
        3: f"/mnt/gpfs01/lsf-workspace/tialab-simon/output/gland_graphs/model_output/clam/{dataset_name}/fold3/results.csv",
        } 
    gt_files = {
        "uhcw": "/mnt/gpfs01/lsf-workspace/tialab-simon/graph_data/cobi/development_info_no_spiro_melanosis.csv",
        "colchester": "/mnt/gpfs01/lsf-workspace/tialab-simon/graph_data/colchester/test_info_colchester_revised.csv",
        "south_warwick": "/mnt/gpfs01/lsf-workspace/tialab-simon/graph_data/south_warwick/test_info_south_warwick_revised.csv",
        "imp": "/mnt/gpfs01/lsf-workspace/tialab-simon/graph_data/imp/test_info_imp_revised.csv"
    }
    gt = gt_files[dataset_name]

    all_results = [results_iguana, results_idars_avg, results_clam, results_rf]
    model_names = ['IGUANA', 'IDaRS', 'CLAM', 'Gland RF']
    color_list = ['blue', 'green', 'red', '#F19A07']
    
    for curve_type in ["roc", "pr"]:
        #* Get ROC and PR plots
        save_names = [f"{curve_type}_{dataset_name}", f"{curve_type}_zoom_{dataset_name}"]
        if curve_type == "roc":
            axes_lims_list = [[[-0.05, 1.05], [-0.05, 1.05]], [[0.0, 0.3], [0.70, 1.0]]]
        else:
            axes_lims_list = [[[-0.05, 1.05], [-0.05, 1.05]], [[0.7, 1.0], [0.7, 1.0]]]
        show_legend_list = [True, False]
        axes_show_list = [[True,True,True,True], [True,True,True,True]]
        ticks_visible_list = [True, False]
        alpha_list = [0.0, 0.2]
        # iterate twice: 
        # 1) entire graph, 2) zoomed in region
        for i in range(2):
            save_name = save_names[i]
            axes_lims = axes_lims_list[i]
            show_legend = show_legend_list[i]
            axes_show = axes_show_list[i]
            ticks_visible = ticks_visible_list[i]
            alpha = alpha_list[i]
            fig, ax = plt.subplots(figsize=(4.5,4.5))
            for idx, results in enumerate(all_results):
                score_list = []
                tp_list = []
                mean_fp = np.linspace(0, 1, 100)
                for fold_nr, results_path in results.items():
                    results_load = pd.read_csv(results_path)
                    scores = results_load['score']
                    wsi_names = results_load['wsi_name']
                    labels, scores, _ = get_labels_scores(wsi_names, scores, gt)
                    
                    if curve_type == "roc":
                        fp, interp_tp, roc_auc = get_vis_roc(
                            labels,
                            scores,
                            mean_fp,
                            )
                        score_list.append(roc_auc)
                    else:
                        fp, interp_tp, avg_pr = get_vis_pr(
                            labels,
                            scores,
                            mean_fp,
                            )
                        score_list.append(avg_pr)
                    tp_list.append(interp_tp)
                
                # draw the curve with confidence intervals
                ax = plot_curve(
                    tp_list,
                    score_list,
                    mean_fp,
                    ax,
                    model_name= model_names[idx],
                    color=color_list[idx],
                    alpha=alpha,
                    lims=axes_lims,
                    show_legend=show_legend,
                    axes_show=axes_show,
                    ticks_visible=ticks_visible,
                    mode=curve_type
                )
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{save_name}.png")

    cat_list = []
    val_list = []
    model_list = []
    colors = []
    spec_all = []
    spec_std = []
    for idx, results in enumerate(all_results):
        model_name = model_names[idx]
        spec_97_list = []
        spec_98_list = []
        spec_99_list = []
        for fold_nr, results_path in results.items():
            results_load = pd.read_csv(results_path)
            scores = results_load['score']
            wsi_names = results_load['wsi_name']
            labels, scores, _ = get_labels_scores(wsi_names, scores, gt)

            spec_95, spec_97, spec_98, spec_99, spec_100 = get_sens_spec_metrics(labels, scores)     

            spec_97_list.append(spec_97)
            spec_98_list.append(spec_98)
            spec_99_list.append(spec_99)

            val_list.append(spec_97)
            val_list.append(spec_98)
            val_list.append(spec_99)
            model_list.extend([model_name]*3)
            colors.extend(['blue', 'green', 'red', '#F19A07'])
        spec_all.append([np.mean(spec_97_list), np.mean(spec_98_list), np.mean(spec_99_list)])
        spec_std.append([np.std(spec_97_list), np.std(spec_98_list), np.std(spec_99_list)])
        
        cat_list.extend(['Sensitivity = 0.97', 'Sensitivity = 0.98', 'Sensitivity = 0.99']*3)
    
    data = {
        "IGUANA": [spec_all[0], spec_std[0]],
        "IDaRS": [spec_all[1], spec_std[1]],
        "CLAM": [spec_all[2], spec_std[2]],
        "Gland RF": [spec_all[3], spec_std[3]],
    }

    fig, ax = plt.subplots(figsize=(4.5,4.5))
    bar_plot(ax, data, colors=['blue', 'green', 'red', '#F19A07'], total_width=.8, single_width=.9)
    plt.xticks([0, 1, 2], ['0.97', '0.98', '0.99'], # sensitivity on the x-axis
       rotation=0)  # Set text labels and properties.
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('')
    plt.ylabel('% Reduction in Normal Slide Reviews (Specificity)', fontsize=11) # same as specificity
    plt.xlabel('Sensitivity', fontsize=14)
    for _,s in ax.spines.items():
        s.set_linewidth(2)
        s.set_color('black')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/sens_spec_bar_{dataset_name}.png")
    

    # create plot indicating the clinical utility of IGUANA - i.e the number of 
    # slides that require review to obtain a certain performance
    fig, ax = plt.subplots(figsize=(7,7))
    min_list = []
    for idx, results in enumerate(all_results):
        if idx == 0:
            color = color_list[idx]
            model_name = model_names[idx]
            x_list = []
            y_list = []
            for fold_nr, results_path in results.items():
                results_load = pd.read_csv(results_path)
                scores = results_load['score']
                wsi_names = results_load['wsi_name']
                labels, scores, prop_normal = get_labels_scores(wsi_names, scores, gt)

                M = screening_info(labels, scores, npts=100)
                
                x_list.append(M[:,-1])
                y_list.append(M[:,-2])
        
            x_mean = np.mean(x_list, axis=0)
            y_mean = np.mean(y_list, axis=0)

            pos = np.argmin(abs(y_mean-0.9))
            x_min = x_mean[pos]
            min_list.append(x_min)
            ax.plot(
                x_mean,
                y_mean,
                color=color,
                lw=3,
                label=f"{model_name}",
                alpha=0.8
                )
            
            std_y = np.std(y_list, axis=0)
            y_upper = np.minimum(y_mean + std_y, 1)
            y_lower = np.maximum(y_mean - std_y, 0)
            ax.fill_between(
                x_mean,
                y_lower,
                y_upper,
                color=color,
                alpha=alpha,
            )

    # set axis limits
    xlim_min = min(min_list)
    if xlim_min > 1 - prop_normal:
        xlim_min = 1 - prop_normal - 0.02
    ax.set(xlim=[xlim_min,1], ylim=[0.9,1])
    
    plt.axvline(x=1 - prop_normal, color='black', ls=':')
    plt.axhline(y=0.99, color='black')
    legend_elements = [
        Line2D([0], [0], color='blue', label='IGUANA', lw=2),
        Line2D([0], [0], color='white', label=''),
        Line2D([0], [0], color='red', label='CLAM', lw=2),
        Line2D([0], [0], color='black', label='99% Sensitivity Target'),
        Line2D([0], [0], color='green', label='IDaRS', lw=2),
        Line2D([0], [0], color='black', label='Abnormal Proportion', ls=':'),
        Line2D([0], [0], color='#F19A07', label='Gland RF', lw=2),
        Line2D([0], [0], color='white', label=''),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.3), fancybox=False, shadow=False, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.xlabel('Proportion of Slides Reviewed')
    plt.ylabel('Sensitivity')
    for _,s in ax.spines.items():
        s.set_linewidth(2)
        s.set_color('black')
    plt.tight_layout()
    # plot inspired from Campanella et al. (2019)
    plt.savefig(f"{save_dir}/campanella_{dataset_name}.png", dpi=300)
            
