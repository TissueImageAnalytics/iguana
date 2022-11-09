<p align="center">
  <img src="doc/iguana.png">
</p>

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
- `explainer`: utility scripts and functions for explanation method.
- `metrics`: utility scripts and functions for computing metrics/statistics.
- `misc`: miscellaneous scripts and functions.
- `models`: scripts relating to defining the model, the hyperparameters and I/O configuration.
- `run_utils`: main engine and callbacks.

## Graph Construction and Training
Insert instructions on how to extract features and train the model

## Inference and Explanation

Insert general overview of inference and explanation

- Run model inference: `python run_infer.py`

- Get node explanations: `python run_explainer.py --node`

- Get feature explanations: `python run_explainer.py --feature`

- Get WSI-level explanations: `python run_explainer.py --wsi`

Note, node and feature explanations must have been run before triggering wsi explanation.

## Interactive Demo
We have made an interactive demo to help visualise the output of our model. Note, this is not optimised for mobile phones and tablets. 

Check out the demo [here](https://iguana.dcs.warwick.ac.uk). 

In the demo, we provide multiple examples of WSI-level results. By default, glands are coloured by their node explanation score, indicating how much they contribute to the slide being predicted as abnormal. Glands can also be coloured by a specific feature using the drop-down menu on the right hand side.

As you zoom in, smaller objects such as lumen and nuclei will become visible. These are accordingly coloured by their predicted class. For example, epithelial cells are coloured green and lymphocytes red.

Each histological object can be toggled on/off by clicking the appropriate buton on the right hand side. Also, the colours and the opacity can be altered. 

To see which histological features are contributing to glands being flagged as abnormal, hover over the corresponding node. To view these nodes, toggle the graph on at the bottom-right of the screen.

## To DO
- [x] Inference code 
- [ ] Training code 
- [ ] Notebooks