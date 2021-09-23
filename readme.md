# Global-context aware generative protein design

This repository contains the Pytorch implementation code for paper:

Global-context aware generative protein design

## Dependencies 
* torch == 1.8.0 (with suitable CUDA and CuDNN version)
* scikit-learn == 0.24.2
* numpy
* argparse
* tqdm

## Overview

* `dataset/` contains a script for downloading datasets.
* `experiments/` contains a pretrained model and its hyperparameter configure file.
* `model/` contains code for building models including `structGNN`, `structTrans`, `GCA`. 

## Install

This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:

```
  conda env create -f environment.yml
  conda activate gca_protein_design
```

## Usage

To download the dataset in need, simply run `download_dataset.sh` file in `./dataset` folder:

```
bash download_dataset.sh
```

To validate the reported results, please open `reproduce.ipynb` and execute the blocks inside.

`model_param.json` is generated for every experiment, users can reproduce the experiment with the same setting as the trained one by the following python code:

```
import json
import argparse

config = json.load(open(svpath + 'model_param.json','r'))
exp = Exp(argparse.Namespace(**config))
```

To train a model from scratch by yourself, users can run `main.py` with optional arguments. For example, we would like to run model GCA with 10 epochs:

```
python main.py --model-type gca --epochs 10
```

More optional arguments are available in `parser.py`.

## Contact

If you have any questions, feel free to contact us through email (tancheng@westlake.edu.cn). Enjoy!