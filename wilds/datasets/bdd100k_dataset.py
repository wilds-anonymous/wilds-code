import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from wilds.common.metrics.all_metrics import MultiTaskAccuracy
from wilds.datasets.wilds_dataset import WILDSDataset


class BDD100KDataset(WILDSDataset):
    """
    The BDD100K-wilds driving dataset.
    This is a modified version of the original BDD100K dataset.
    This dataset is not part of the official WILDS benchmark.
    We provide it for convenience and to reproduce observations discussed in the WILDS paper.

    Supported `split_scheme`:
        'official', 'timeofday' (equivalent to 'official'), or 'location'

    Input (x):
        1280x720 RGB images of driving scenes from dashboard POV.

    Output (y):
        y is a 9-dimensional binary vector that is 1 at index i if
        BDD100KDataset.CATEGORIES[i] is present in the image and 0 otherwise.

    Metadata:
        If `split_scheme` is 'official' or 'timeofday', each data point is
        annotated with a time of day from BDD100KDataset.TIMEOFDAY_SPLITS.
        If `split_scheme` is 'location' each data point is annotated with a
        location from BDD100KDataset.LOCATION_SPLITS

    Website:
        https://bdd-data.berkeley.edu/

    Original publication:
        @InProceedings{bdd100k,
            author = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen,
                      Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
            title = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
            booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
            month = {June},
            year = {2020}
        }

    License (original text):
        Copyright ©2018. The Regents of the University of California (Regents). All Rights Reserved.
        Permission to use, copy, modify, and distribute this software and its documentation for educational, research, and
        not-for-profit purposes, without fee and without a signed licensing agreement; and permission use, copy, modify and
        distribute this software for commercial purposes (such rights not subject to transfer) to BDD member and its affiliates,
        is hereby granted, provided that the above copyright notice, this paragraph and the following two paragraphs appear in
        all copies, modifications, and distributions. Contact The Office of Technology Licensing, UC Berkeley, 2150 Shattuck
        Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-7201, otl@berkeley.edu,
        http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.
        IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
        INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN ADVISED
        OF THE POSSIBILITY OF SUCH DAMAGE.
        REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
        AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED
        "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
    """

    CATEGORIES = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'rider',
                  'traffic light', 'traffic sign', 'truck']
    TIMEOFDAY_SPLITS = ['daytime', 'night', 'dawn/dusk', 'undefined']
    LOCATION_SPLITS = ['New York', 'California']

    def __init__(self, root_dir='data', download=False, split_scheme='official'):
        self._dataset_name = 'bdd100k'
        self._version = '1.0'
        self._original_resolution = (1280, 720)
        self._download_url = ''
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self.root = Path(self.data_dir)

        if split_scheme in ('official', 'timeofday'):
            split_to_load = 'timeofday'
        elif split_scheme == 'location':
            split_to_load = 'location'
        else:
            raise ValueError("For BDD100K, split scheme should be 'official', "
                             "'timeofday', or 'location'.")
        self._split_scheme = split_scheme
        train_data_df = pd.read_csv(self.root / f'{split_to_load}_train.csv')
        val_data_df = pd.read_csv(self.root / f'{split_to_load}_val.csv')
        test_data_df = pd.read_csv(self.root / f'{split_to_load}_test.csv')
        self._image_array = []
        self._split_array, self._y_array, self._metadata_array = [], [], []

        for i, df in enumerate([train_data_df, val_data_df, test_data_df]):
            self._image_array.extend(list(df['image'].values))
            labels = [list(df[cat].values) for cat in self.CATEGORIES]
            labels = list(zip(*labels))
            self._split_array.extend([i] * len(labels))
            self._y_array.extend(labels)
            self._metadata_array.extend(list(df['group'].values))
        self._y_size = len(self.CATEGORIES)
        self._metadata_fields = [split_to_load]
        self._split_array = np.array(self._split_array)
        self._y_array = torch.tensor(self._y_array, dtype=torch.float)
        self._metadata_array = torch.tensor(self._metadata_array,
                                            dtype=torch.long).unsqueeze(1)
        split_names = (self.TIMEOFDAY_SPLITS if split_to_load == 'timeofday'
                       else self.LOCATION_SPLITS)
        self._metadata_map = {split_to_load: split_names}
        self._metric = MultiTaskAccuracy()

    def get_input(self, idx):
        img = Image.open(self.root / 'images' / self._image_array[idx])
        return img

    def eval(self, y_pred, y_true, metadata):
        results = self._metric.compute(y_pred, y_true)
        results_str = (f'{self._metric.name}: '
                       f'{results[self._metric.agg_metric_field]:.3f}\n')
        return results, results_str
