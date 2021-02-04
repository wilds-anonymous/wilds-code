import os
import torch
import pandas as pd
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy

class CivilCommentsDataset(WILDSDataset):
    """
    The CivilComments-wilds toxicity classification dataset.
    This is a modified version of the original CivilComments dataset.

    Supported `split_scheme`:
        'official'

    Input (x):
        A comment on an online article, comprising one or more sentences of text.

    Label (y):
        y is binary. It is 1 if the comment was been rated as toxic by a majority of the crowdworkers who saw that comment, and 0 otherwise.

    Metadata:
        Each comment is annotated with the following binary indicators:
            - male
            - female
            - LGBTQ
            - christian
            - muslim
            - other_religions
            - black
            - white
            - identity_any
            - severe_toxicity
            - obscene
            - threat
            - insult
            - identity_attack
            - sexual_explicit

    Website:
        https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification

    Original publication:
        @inproceedings{borkan2019nuanced,
          title={Nuanced metrics for measuring unintended bias with real data for text classification},
          author={Borkan, Daniel and Dixon, Lucas and Sorensen, Jeffrey and Thain, Nithum and Vasserman, Lucy},
          booktitle={Companion Proceedings of The 2019 World Wide Web Conference},
          pages={491--500},
          year={2019}
        }

    License:
        This dataset is in the public domain and is distributed under CC0.
        https://creativecommons.org/publicdomain/zero/1.0/
    """

    def __init__(self, root_dir='data', download=False, split_scheme='official'):
        self._dataset_name = 'civilcomments'
        self._version = '1.0'
        self._download_url = ''
        self._compressed_size = 90_644_480
        self._data_dir = self.initialize_data_dir(root_dir, download)

        # Read in metadata
        self._metadata_df = pd.read_csv(
            os.path.join(self._data_dir, 'all_data_with_identities.csv'),
            index_col=0)

        # Get the y values
        self._y_array = torch.LongTensor(self._metadata_df['toxicity'].values >= 0.5)
        self._y_size = 1
        self._n_classes = 2

        # Extract text
        self._text_array = list(self._metadata_df['comment_text'])

        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        # metadata_df contains split names in strings, so convert them to ints
        for split in self.split_dict:
            split_indices = self._metadata_df['split'] == split
            self._metadata_df.loc[split_indices, 'split'] = self.split_dict[split]
        self._split_array = self._metadata_df['split'].values

        # Extract metadata
        self._identity_vars = [
            'male',
            'female',
            'LGBTQ',
            'christian',
            'muslim',
            'other_religions',
            'black',
            'white'
        ]
        self._auxiliary_vars = [
            'identity_any',
            'severe_toxicity',
            'obscene',
            'threat',
            'insult',
            'identity_attack',
            'sexual_explicit'
        ]

        self._metadata_array = torch.cat(
            (
                torch.LongTensor((self._metadata_df.loc[:, self._identity_vars] >= 0.5).values),
                torch.LongTensor((self._metadata_df.loc[:, self._auxiliary_vars] >= 0.5).values),
                self._y_array.reshape((-1, 1))
            ),
            dim=1
        )
        self._metadata_fields = self._identity_vars + self._auxiliary_vars + ['y']

        self._eval_groupers = [
            CombinatorialGrouper(
                dataset=self,
                groupby_fields=[identity_var, 'y'])
            for identity_var in self._identity_vars]
        self._metric = Accuracy()

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        return self._text_array[idx]

    def eval(self, y_pred, y_true, metadata):
        results = {
            **self._metric.compute(y_pred, y_true),
        }
        results_str = f"Average {self._metric.name}: {results[self._metric.agg_metric_field]:.3f}\n"
        # Each eval_grouper is over label + a single identity
        # We only want to keep the groups where the identity is positive
        # The groups are:
        #   Group 0: identity = 0, y = 0
        #   Group 1: identity = 1, y = 0
        #   Group 2: identity = 0, y = 1
        #   Group 3: identity = 1, y = 1
        # so this means we want only groups 1 and 3.
        worst_group_metric = None
        for identity_var, eval_grouper in zip(self._identity_vars, self._eval_groupers):
            g = eval_grouper.metadata_to_group(metadata)
            group_results = {
                **self._metric.compute_group_wise(y_pred, y_true, g, eval_grouper.n_groups)
            }
            results_str += f"  {identity_var:20s}"
            for group_idx in range(eval_grouper.n_groups):
                group_str = eval_grouper.group_field_str(group_idx)
                if f'{identity_var}:1' in group_str:
                    group_metric = group_results[self._metric.group_metric_field(group_idx)]
                    group_counts = group_results[self._metric.group_count_field(group_idx)]
                    results[f'{self._metric.name}_{group_str}'] = group_metric
                    results[f'count_{group_str}'] = group_counts
                    if f'y:0' in group_str:
                        label_str = 'non_toxic'
                    else:
                        label_str = 'toxic'
                    results_str += (
                        f"   {self._metric.name} on {label_str}: {group_metric:.3f}"
                        f" (n = {results[f'count_{group_str}']:6.0f}) "
                    )
                    if worst_group_metric is None:
                        worst_group_metric = group_metric
                    else:
                        worst_group_metric = self._metric.worst(
                            [worst_group_metric, group_metric])
            results_str += f"\n"
        results[f'{self._metric.worst_group_metric_field}'] = worst_group_metric
        results_str += f"Worst-group {self._metric.name}: {worst_group_metric:.3f}\n"

        return results, results_str
