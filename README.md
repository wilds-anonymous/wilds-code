<p align='center'>
  <img width='40%' src='https://wilds-anonymous.github.io/WILDS_cropped.png' />
</p>

--------------------------------------------------------------------------------

## Anonymized repo for review
This repo has been anonymized for review. As the dataset download URLs are not anonymous, we have therefore removed the option to automatically download the datasets. The example scripts will not be able to run without the datasets. We have also removed pip install instructions for the public, non-anonymized package.

## Overview
WILDS is a benchmark of in-the-wild distribution shifts spanning diverse data modalities and applications, from tumor identification to wildlife monitoring to poverty mapping.

The WILDS package contains:
1. Data loaders that automatically handle data downloading, processing, and splitting, and
2. Dataset evaluators that standardize model evaluation for each dataset.

In addition, the example scripts contain default models, allowing new algorithms to be easily added and run on all of the WILDS datasets.


## Installation

This anonymized repository can only be installed from source.

```bash
git clone https://github.com/wilds-anonymous/wilds-code.git
cd wilds
pip install -e .
```

### Requirements
- numpy>=1.19.1
- pandas>=1.1.0
- pillow>=7.2.0
- torch>=1.7.0
- tqdm>=4.53.0
- pytz>=2020.4
- outdated>=0.2.0
- ogb>=1.2.3
- torch-scatter>=2.0.5
- torch-geometric>=1.6.1

Installing with `pip` will  will check for all of these requirements except for the `torch-scatter` and `torch-geometric` packages, which require a [quick manual install](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-binaries).


### Default models
After installing the WILDS package, you can use the scripts in `examples/` to train default models on the WILDS datasets.

To run these scripts, you will need to install these additional dependencies:

- torchvision>=0.8.1
- transformers>=3.5.0

All baseline experiments in the paper were run on Python 3.8.5 and CUDA 10.1.

## Usage
### Default models
In the `examples/` folder, we provide a set of scripts that we used to train models on the WILDS package. These scripts are configured with the default models and hyperparameters that we used for all of the baselines described in our paper. All baseline results in the paper can be easily replicated with commands like:

```bash
cd examples
python run_expt.py --dataset iwildcam --algorithm ERM --root_dir data
python run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data
```

The scripts are set up to facilitate general-purpose algorithm development: new algorithms can be added to `examples/algorithms` and then run on all of the WILDS datasets using the default models.

The first time you run these scripts, you might need to download the datasets. You can do so with the `--download` argument, for example:
```
python run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --download
```

### Data loading

The WILDS package provides a simple, standardized interface for all datasets in the benchmark.
This short Python snippet covers all of the steps of getting started with a WILDS dataset, including dataset download and initialization, accessing various splits, and preparing a user-customizable data loader.

```py
>>> from wilds.datasets.iwildcam_dataset import IWildCamDataset
>>> from wilds.common.data_loaders import get_train_loader
>>> import torchvision.transforms as transforms

# Load the full dataset, and download it if necessary
>>> dataset = IWildCamDataset(download=True)

# Get the training set
>>> train_data = dataset.get_subset('train',
...                                 transform=transforms.Compose([transforms.Resize((224,224)),
...                                                               transforms.ToTensor()]))

# Prepare the standard data loader
>>> train_loader = get_train_loader('standard', train_data, batch_size=16)

# Train loop
>>> for x, y_true, metadata in train_loader:
...   ...
```

The `metadata` contains information like the domain identity, e.g., which camera a photo was taken from, or which hospital the patient's data came from, etc.

### Domain information
To allow algorithms to leverage domain annotations as well as other
groupings over the available metadata, the WILDS package provides `Grouper` objects.
These `Grouper` objects extract group annotations from metadata, allowing users to
specify the grouping scheme in a flexible fashion.

```py
>>> from wilds.common.grouper import CombinatorialGrouper

# Initialize grouper, which extracts domain information
# In this example, we form domains based on location
>>> grouper = CombinatorialGrouper(dataset, ['location'])

# Train loop
>>> for x, y_true, metadata in train_loader:
...   z = grouper.metadata_to_group(metadata)
...   ...
```

The `Grouper` can be used to prepare a group-aware data loader that, for each minibatch, first samples a specified number of groups, then samples examples from those groups.
This allows our data loaders to accommodate a wide array of training algorithms,
some of which require specific data loading schemes.

```py
# Prepare a group data loader that samples from user-specified groups
>>> train_loader = get_train_loader('group', train_data,
...                                 grouper=grouper,
...                                 n_groups_per_batch=2,
...                                 batch_size=16)
```

### Evaluators

The WILDS package standardizes and automates evaluation for each dataset.
Invoking the `eval` method of each dataset yields all metrics reported in the paper and on the leaderboard.

```py
>>> from wilds.common.data_loaders import get_eval_loader

# Get the test set
>>> test_data = dataset.get_subset('test',
...                                 transform=transforms.Compose([transforms.Resize((224,224)),
...                                                               transforms.ToTensor()]))

# Prepare the data loader
>>> test_loader = get_eval_loader('standard', test_data, batch_size=16)

# Get predictions for the full test set
>>> for x, y_true, metadata in test_loader:
...   y_pred = model(x)
...   [accumulate y_true, y_pred, metadata]

# Evaluate
>>> dataset.eval(all_y_pred, all_y_true, all_metadata)
{'recall_macro_all': 0.66, ...}
```


