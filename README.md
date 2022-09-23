# Hippocampus Segmentation Model Comparison

## Introduction

This directory contains the code used for Toh You Sheng's DSA4199 Honours Thesis titled "A comparison of 2D and 3D convolution models for the volumetric segmentation of the human hippocampus". In this thesis, two deep learning models, 3D U-Net and 2D U-Seg-Net are compared for their performance and training efficiency when used for segmenting the human hippocampus from brain MRI. This code is entirely written in Python 3.9, with heavy use of the PyTorch framework for the implementation of the models. Experiments were performed using rented Paperspace Gradient notebooks with access to a NVIDIA RTX A4000 GPU. 

If any difficulties or problems are encountered in the execution of the following code, please contact me at <yousheng.toh@gmail.com>


## Environment Setup

To use the code written, I highly recommend setting up a conda environment to install all packages and their dependencies. Instructions on how to set up a conda environment using miniconda can be found [here](https://docs.conda.io/en/latest/miniconda.html)

Included with the code is a "environment.yml" file, which can be used to create the necessary environment from scratch using the following command:

 `conda env create -f environment.yml`

To maintain the code, I have tried to keep this environment.yml file as updated as possible. However, updates may have been made to some of these packages from the submission of this code till this current point in time. Therefore, some of the packages in the environment.yml file may not work from time to time. 

If that is the case, please note that these are the packages needed to be installed (via conda and pip respectively, some packages do not support conda installs).

Using conda:
- numpy
- pandas
- matplotlib
- jupyterlab
- scikit-learn
- SimpleITK (-c <https://conda.anaconda.org/simpleitk>)
- pytorch
- torchvision
- cudatoolkit=11.3
- tqdm

Using pip:
- nibabel
- nilearn
- intensity-normalization
- torchio
- torchinfo

I strongly suggest that this environment be set up on a computer or server with GPU access (a 16GB NVIDIA RTX A4000 was used for this thesis). This is because some parts of the code and especially the training of the models require the computational efficiency of GPUs.


## Code Summary

Each folder and file in the code folder is briefly described below:

* Preprocessing.ipynb: Jupyter notebook that contains a brief walkthrough of the preprocessing steps taken to get the MRI data ready for the experiments. However, running of this is optional, as the prepared data is already available in the znorm_cropped_imgs folder.

* 3DUNet.ipynb: Jupyter notebook that contains a brief walkthrough of the experiment process for 3D U-Net models

* EnsembleUSegNet.ipynb: Jupyter notebook that contains a brief walkthrough of the experiment process for 2D U-Seg-Net models and the EnsembleUSegNet models

* data_utils: contains two Python modules, dataset.py and preprocessing.py
	* dataset.py: defines the PyTorch dataset object to parse the MRI data in this thesis.
	* preprocessing.py: contains the scripts for preprocessing of MRI data

* model_utils: contains two Python modules, metrics.py and train_eval.py
	* metrics.py: contains the code for computing various metrics such as Dice Similarity Coefficient
	* train_eval.py: contains the main functions for performing training, cross validation and evaluation of the models

* models: contains two Python modules, unet3d.py and usegnet.py
	* unet3d.py: contains the PyTorch implementation of the 3D U-Net architecture
	* usegnet.py: contains the PyTorcch implementation of the 2D U-Seg-Net architecture

* znorm_cropped_imgs: contains the brain MRI volumes and hippocampus segmentation masks in the form of NIfTI files, after processing the HarP dataset

* harp_metadata: contains background information about the subjects which the brain MRIs are taken from, such as cognitive function etc.

* saved_models: contains the saved models in the form of .pth files, which can be directly loaded into the PyTorch model objects

* saved_histories: contains compressed dictionaries of numpy arrays (.npz files) that store information such as training loss and duration for each model during our experiments


