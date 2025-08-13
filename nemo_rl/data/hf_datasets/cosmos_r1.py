## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Cosmos R1 dataset
from typing import Any, Optional
import json

from datasets import Dataset, load_dataset, concatenate_datasets

from nemo_rl.data.interfaces import TaskDataSpec
from PIL import Image
import io
import base64
import os

# these subsets are defined in the dataset, we will combine all subsets into a single dataset
cosmos_common_subsets = ['agibot', 'bridgev2', 'holoassist', 'robovqa']
cosmos_train_subsets_extra = []
cosmos_val_subsets_extra = ['robofail']

def validate_video_paths(ds: Dataset):
    '''
    for all rows in the dataset, verify if that video exists.
    If it doesn't exist, download the video depending on the subset
    '''
    def download_video(video_path, subset):
        video_name = video_path.split('/')[-1]
        if subset == 'agibot':
            ds = load_dataset("agibot-world/AgiBotWorld-Alpha")['train']
            
        elif subset == 'robovqa':
            url = f"https://storage.cloud.google.com/gdm-robovqa/videos/{video_name}"


    
    for row in ds:
        video_path = row['video']
        subset = row['subset']
        if not os.path.exists(video_path):
            print(f"Video {video_path} does not exist, downloading...")
            download_video(video_path, subset)


def prepare_cosmos_r1_reason_dataset(split: str = "default", task_name: str = "cosmos_r1_reason", video_root_dir: Optional[str] = None):
    ''' 
    This function performs the following operations:
    1. Loads the training and validation datasets
        For training, we use `nvidia/Cosmos-Reason1-RL-Dataset` and for validation, we use `nvidia/Cosmos-Reason1-Benchmark`. 
    2. For each subset, we add a field 'subset' to the dataset, which is the subset name
    3. For each video name, we prepend the video_root_dir/subset_name/ to the video name
    4. we concatenate all the data subsets

    '''
    assert split in ['default'], f"Invalid split: {split}. Please use 'default'."
    assert video_root_dir is not None, "video_root_dir must be provided"

    def generate_dataset_helper(dataset_name, subsets,):
        ''' helper function to load multiple subsets from the dataset, add a new column, and modify the video path '''
        ds = [load_dataset(dataset_name, subset)['benchmark'] for subset in subsets]
        for ds, subset in zip(ds, subsets):
            ds = ds.add_column("subset", [subset] * len(ds))
            ds = ds.map(lambda datum: {**datum, "video": os.path.join(video_root_dir, subset, datum['video'])})
        ds = concatenate_datasets(ds)
        return ds

    train_ds = generate_dataset_helper("nvidia/Cosmos-Reason1-RL-Dataset", cosmos_common_subsets + cosmos_train_subsets_extra, )
    val_ds = generate_dataset_helper("nvidia/Cosmos-Reason1-Benchmark", cosmos_common_subsets + cosmos_val_subsets_extra, )

    validate_video_paths(train_ds)
    validate_video_paths(val_ds)
    return {
        'train': train_ds,
        'validation': val_ds,
    }


class CosmosR1ReasonDataset:
    def __init__(self, split: str = "default", prompt_file: Optional[str] = None, video_root_dir: Optional[str] = None, task_name: str = "cosmos_r1_reason"):
        """
        Simple wrapper around the Cosmos R1 Reason dataset.

        Args:
            split: The split of the dataset to use.
            prompt_file: The file containing the prompt for the dataset.
            task_name: The name of the task.
        """
        if split not in ['default']:
            raise ValueError(f"Invalid split: {split}. Please use 'default'.")
        
        self.formatted_ds = prepare_cosmos_r1_reason_dataset(split=split, task_name=task_name, video_root_dir=video_root_dir)
        self.task_spec = TaskDataSpec(
            task_name="Cosmos R1 Reason",
            prompt_file=prompt_file,
        )

