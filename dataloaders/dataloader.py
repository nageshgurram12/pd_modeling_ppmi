# -*- coding: utf-8 -*-
import torch.utils.data as data
import os.path

from mypath import Path
from preprocessing import preprocess_data

class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        
        final_file = Path.get_path('merged_and_padded')
        if not os.path.exists(final_file):
            preprocess_data(pad_missing_visits=True)
        
        self.