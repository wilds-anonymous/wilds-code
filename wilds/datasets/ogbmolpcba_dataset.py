import os
import torch
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.utils.url import download_url
from torch_geometric.data.dataloader import Collater as PyGCollater

class OGBPCBADataset(WILDSDataset):
    """
    The OGB-molpcba dataset.
    This dataset is directly adopted from Open Graph Benchmark, and originally curated by MoleculeNet.

    Supported `split_scheme`:
        'official' or 'scaffold', which are equivalent

    Input (x):
        Molecular graphs represented as Pytorch Geometric data objects

    Label (y):
        y represents 128-class binary labels.

    Metadata:
        - scaffold
            Each molecule is annotated with the scaffold ID that the molecule is assigned to.

    Website:
        https://ogb.stanford.edu/docs/graphprop/#ogbg-mol

    Original publication:
        @article{hu2020ogb,
            title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
            author={W. {Hu}, M. {Fey}, M. {Zitnik}, Y. {Dong}, H. {Ren}, B. {Liu}, M. {Catasta}, J. {Leskovec}},
            journal={arXiv preprint arXiv:2005.00687},
            year={2020}
        }

        @article{wu2018moleculenet,
            title={MoleculeNet: a benchmark for molecular machine learning},
            author={Z. {Wu}, B. {Ramsundar}, E. V {Feinberg}, J. {Gomes}, C. {Geniesse}, A. S {Pappu}, K. {Leswing}, V. {Pande}},
            journal={Chemical science},
            volume={9},
            number={2},
            pages={513--530},
            year={2018},
            publisher={Royal Society of Chemistry}
        }

    License:
        This dataset is distributed under the MIT license.
        https://github.com/snap-stanford/ogb/blob/master/LICENSE
    """

    def __init__(self, root_dir='data', download=False, split_scheme='official'):
        # internally call ogb package
        self.ogb_dataset = PygGraphPropPredDataset(name = 'ogbg-molpcba', root = root_dir)

        # set variables
        self._dataset_name = 'ogbg-molpcba'
        self._data_dir = self.ogb_dataset.root
        if split_scheme=='official':
            split_scheme = 'scaffold'
        self._split_scheme = split_scheme
        self._y_type = 'float' # although the task is binary classification, the prediction target contains nan value, thus we need float
        self._y_size = self.ogb_dataset.num_tasks
        self._n_classes = self.ogb_dataset.__num_classes__

        self._split_array = torch.zeros(len(self.ogb_dataset)).long()
        split_idx  = self.ogb_dataset.get_idx_split()
        self._split_array[split_idx['train']] = 0
        self._split_array[split_idx['valid']] = 1
        self._split_array[split_idx['test']] = 2

        self._y_array = self.ogb_dataset.data.y

        self._metadata_fields = ['scaffold']

        metadata_file_path = os.path.join(self.ogb_dataset.root, 'raw', 'scaffold_group.npy')
        if not os.path.exists(metadata_file_path):
            download_url('', os.path.join(self.ogb_dataset.root, 'raw'))
        self._metadata_array = torch.from_numpy(np.load(metadata_file_path)).reshape(-1,1).long()
        self._collate = PyGCollater(follow_batch=[])

        self._metric = Evaluator('ogbg-molpcba')

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        return self.ogb_dataset[int(idx)]

    def eval(self, y_pred, y_true, metadata):
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        results = self._metric.eval(input_dict)

        return results, f"Average precision: {results['ap']:.3f}\n"
