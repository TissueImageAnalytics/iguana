<p align="center">
  <img src="doc/iguana.png">
</p>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-orange.svg)](https://www.gnu.org/licenses/gpl-3.0)
  <a href="#cite-this-repository"><img src="https://img.shields.io/badge/Cite%20this%20repository-BibTeX-brightgreen" alt="DOI"></a> <a href="https://doi.org/10.1101/2022.10.17.22279804"><img src="https://img.shields.io/badge/DOI-10.1101%2F2022.10.17.22279804-blue" alt="DOI"></a>
<br>

# Interpretable Gland-Graph Networks using a Neural Aggregator

IGUANA is a graph neural network built for colon biopsy screening. IGUANA represents a whole-slide image (WSI) as a graph built with nodes on top of glands in the tissue, each node associated with a set of interpretable features. The output of the pipeline is explainable, indicating glands and features that contribute to a WSI being predicted as abnormal. 

For a full description, take a look at our [preprint](https://doi.org/10.1101/2022.10.17.22279804).

## Set Up Environment

```
# create base conda environment
conda env create -f environment.yml

# activate environment
conda activate iguana

# install PyTorch with pip
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html

# install PyTorch Geometric and dependencies
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu102.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.1+cu102.html
pip install torch-geometric
```

## Repository Structure

- `doc`: image files used for rendering the README - not necessary for running the code. 
- `dataloader`: contains code for loading the data to the model.
- `metrics`: utility scripts and functions for computing metrics/statistics.
- `misc`: miscellaneous scripts and functions.
- `models`: scripts relating to defining the model, the hyperparameters and I/O configuration.
- `run_utils`: main engine and callbacks.

## Graph Construction

### Histological Segmentation
As a first step, WSIs need to be processed with Cerberus to perform simultaneous prediction of:
- Nuclear segmentation and classification
- Gland segmentation and classification
- Lumen segmentation
- Tissue type classification

Refer to [this repository](https://github.com/TissueImageAnalytics/cerberus) for details on how to run Cerberus. Note, in this repo we also assume that tissue masks have been extracted before running Cerberus.

### Feature Extraction
To perform feature extraction run `extract-feats.py`. To see a full list of command line arguments, use:

```
python extract_feats.py -h
```

You will see that one of the arguments is `focus_mode`. This denotes the non-epithelial area considered for cell quantification. For the original publication, we considered the lamina propria (`focus_mode=lp`) which is the area surrounding the glands. If `lp` is not selected, then the entire tissue is considered for cell quantification.

### Constructing the Graphs
Next, we convert the features into a format to be used by our graph learning model. Here, we also fill in missing values (denoted by `nan`).

To do this, run:

```
python create_graph.py --input_path=<path> --output_path=<path> --dist_thresh=<n> --data_info=<path>
```

- `input_path`: path to features extracted using `extract_feats.py`.
- `output_path`: path to where graph data will be saved.
- `dist_thresh`: only connect glands if they are within a distance less than this threshold. 
- `data_info`: information regarding the labels and fold info of the data. More info on this is provided when describing [model training](#training-the-model).

The above command converts the data to a dictionary with the following keys:

- `local_feats`: a dictionary where each key denotes the feature name and the corresponding values denote the features. Note, in this dictionary we also include the `obj_id`, which represents the unique identifier of each gland (as originally used in the Cerberus output).
- `edge_index`: 2xN array of indicies showing the presence of edges between nodes.
- `label`: The label of the graph.
- `wsi_name`: Filename of the WSI.


## Inference

To see the full list of command line arguments for inference and explanation, run `python run_infer.py -h` and `python run_explainer.py -h`, respectively. We have also created two bash scripts to make it easier to run the code with the appropriate arguments. As an example, to run model inference enter:

```
python run_infer.py --gpu=<gpu_id> --model_path=<path> --data_dir=<path> --data_info=<path> --stats_dir=<path>
```

You will see above that the `data_info` csv file will need to be incorporated as an argument. This will determine the label and which images to process. By default, the code will process images with values in the fold column equal to 3. If considering a test set, there will be a single 'fold' column named `test_info`. The `fold_nr` and `split_nr` can be added as additional arguments if considering cross validation, which determines the subset of the data from the csv file.


## Interactive Demo
We have made an interactive demo to help visualise the output of our model. Note, this is not optimised for mobile phones and tablets. The demo was built using the TIAToolbox [tile server](https://tia-toolbox.readthedocs.io/en/latest/_autosummary/tiatoolbox.visualization.tileserver.TileServer.html).

Check out the demo [here](https://iguana.dcs.warwick.ac.uk). 

In the demo, we provide multiple examples of WSI-level results. By default, glands are coloured by their node explanation score, indicating how much they contribute to the slide being predicted as abnormal. Glands can also be coloured by a specific feature using the drop-down menu on the right hand side.

As you zoom in, smaller objects such as lumen and nuclei will become visible. These are accordingly coloured by their predicted class. For example, epithelial cells are coloured green and lymphocytes red.

Each histological object can be toggled on/off by clicking the appropriate buton on the right hand side. Also, the colours and the opacity can be altered. 

To see which histological features are contributing to glands being flagged as abnormal, hover over the corresponding node. To view these nodes, toggle the graph on at the bottom-right of the screen.

![demo](https://user-images.githubusercontent.com/20071401/201095785-d7ce01c3-9652-425e-b1cd-38dd22672b81.gif)

## Sample Data and Weights
We have released a small portion of data to allow researchers to get the code running and see how our graph data is structured. We also include two 'data info' csv files - one as if the data is to be used for cross validation and the other as an external test set. Click [here](https://drive.google.com/drive/folders/14MYdB-Acb93L6IdJfHnlWG7M2pkBn53I?usp=sharing) to download the sample dataset.

To download the **IGUANA weights** trained on each fold of the UHCW dataset, click [here](https://drive.google.com/drive/folders/1J78dItPqMZcj2BsM4mf69x61wNAuJwKP?usp=share_link). To get the code running, you will also need the stats info used to standardise the input data. This includes the statistics (mean, mean, etc) of the features and the input node degree. Click [here](https://drive.google.com/drive/folders/1vcbf-9YrtoQUpFvf_l73iy5TupVWEdBF?usp=sharing) to download the stats info.
## License

Code is under a GPL-3.0 license. See the [LICENSE](https://github.com/TissueImageAnalytics/cerberus/blob/master/LICENSE) file for further details.

Model weights are licensed under [Attribution-NonCommercial-ShareAlike 4.0 International](http://creativecommons.org/licenses/by-nc-sa/4.0/). Please consider the implications of using the weights under this license. 

## Cite this repository

```
@article{graham2022screening,
  title={Screening of normal endoscopic large bowel biopsies with artificial intelligence: a retrospective study},
  author={Graham, Simon and Minhas, Fayyaz and Bilal, Mohsin and Ali, Mahmoud and Tsang, Yee Wah and Eastwood, Mark and Wahab, Noorul and Jahanifar, Mostafa and Hero, Emily and Dodd, Katherine and others},
  journal={medRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory Press}
}
```
