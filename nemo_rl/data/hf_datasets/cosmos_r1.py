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
import requests

from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import torch

# these subsets are defined in the dataset, we will combine all subsets into a single dataset
cosmos_common_subsets = ['agibot', 'bridgev2', 'holoassist', 'robovqa']
cosmos_train_subsets_extra = []
cosmos_val_subsets_extra = ['robofail']

def validate_video_paths(ds: Dataset) -> Dataset:
    '''
    for all rows in the dataset, verify if that video exists.
    If it doesn't exist, download the video depending on the subset
    '''
    def download_video(video_path, subset):
        video_name = video_path.split('/')[-1]
        if subset == 'agibot':
            # ds = load_dataset("agibot-world/AgiBotWorld-Alpha")['train']
            # dataset too large, skip for now
            return False 
        
        elif subset == 'bridgev2':
            # dataset too large, skip for now
            return False
            
        elif subset == 'robovqa':
            url = f"https://storage.googleapis.com/gdm-robovqa/videos/{video_name}"
            # save video to `video_path`
            response = requests.get(url)
            with open(video_path, 'wb') as f:
                f.write(response.content)
            return True
        
        elif subset == 'holoassist':
            # dataset too large, skip for now
            return False
        
        elif subset == 'robofail':
            # dataset too large, skip for now
            return False

        return False
    
    # find all failed videos
    failed_videos = []
    for row in ds:
        video_path = row['video']  # this is the path where the video is (supposed to be stored)
        subset = row['subset']
        if not os.path.exists(video_path):
            # print(f"Video {video_path} does not exist, downloading...")
            ret = download_video(video_path, subset)
            if not ret:
                failed_videos.append(video_path)

    # filter out the failed videos
    print(f"Filtering out {len(failed_videos)} failed video downloads")
    ds = ds.filter(lambda row: row['video'] not in failed_videos)
    return ds

def add_task_name(ds: Dataset, task_name: str) -> Dataset:
    '''
    Add a task_name column to the dataset
    '''
    ds = ds.add_column("task_name", [task_name] * len(ds))
    return ds

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

    def generate_dataset_helper(dataset_name: str, subsets: list[str]) -> Dataset:
        ''' helper function to load multiple subsets from the dataset, add a new column, and modify the video path '''
        def get_datadicts(dataset_name, subset):
            dsdict = load_dataset(dataset_name, subset)
            keys = list(dsdict.keys())
            ds = [dsdict[key] for key in keys]
            ds = concatenate_datasets(ds)
            return ds

        all_ds = [get_datadicts(dataset_name, subset) for subset in subsets]
        for i, (ds, subset) in enumerate(zip(all_ds, subsets)):
            ds = ds.add_column("subset", [subset] * len(ds))
            ds = ds.map(lambda datum: {**datum, "video": os.path.join(video_root_dir, subset, datum['video'].replace("clips/", ""))})
            all_ds[i] = ds
        ds = concatenate_datasets(all_ds)
        # create all paths
        paths = set([os.path.dirname(row['video']) for row in ds])
        for path in paths:
            os.makedirs(os.path.join(video_root_dir, path), exist_ok=True)
        # return dataset
        return ds

    train_ds = generate_dataset_helper("nvidia/Cosmos-Reason1-RL-Dataset", cosmos_common_subsets + cosmos_train_subsets_extra)
    val_ds = generate_dataset_helper("nvidia/Cosmos-Reason1-Benchmark", cosmos_common_subsets + cosmos_val_subsets_extra)

    train_ds = add_task_name(validate_video_paths(train_ds), task_name)
    val_ds = add_task_name(validate_video_paths(val_ds), task_name)
    return {
        'train': train_ds,
        'validation': val_ds,
    }


def get_pil_video(video_path: str, target_fps: float = 2.0) -> Image.Image:
    """
    Given a local video path, return a list of PIL.Image frames from the video.
    Uses decord for video decoding.
    """
    frames = []

    vr = VideoReader(video_path, ctx=cpu(0), width=224, height=224)
    total_frames, fps = len(vr), vr.get_avg_fps()
    target_sample_rate = fps / target_fps  # this is the resampling rate
    sample_indices = torch.arange(0, total_frames, target_sample_rate).round().long().tolist()
    if sample_indices[-1] == total_frames:
        sample_indices[-1] -= 1
    print("sample_indices", sample_indices)
    print("total frames", total_frames)
    # breakpoint()
    # get frames, and permute
    frames = vr.get_batch(sample_indices).asnumpy()
    frames = torch.tensor(frames).permute(0, 3, 1, 2)
    print("Returning frames of shape", frames.shape)
    return frames, target_fps


def format_cosmos_r1_reason_dataset(example: dict[str, Any]) -> dict[str, Any]:
    """
    Format the Cosmos R1 Reasoning dataset into an OpenAI-API-like message log.
    
    Args:
        example: Dictionary containing video path and QA pairs
        
    Returns:
        Formatted dictionary with messages in OpenAI-API format
    """
    # Format question and options
    question = example['qa_pairs']['question']
    index2ans = example['qa_pairs']['index2ans']
    sorted_options = [f"({key}) {index2ans[key]}" for key in sorted(index2ans.keys())]
    formatted_qa = f"{question}\n{', '.join(sorted_options)}"
    
    frames, frame_rate = get_pil_video(example['video'])
    user_content = [
        {
            "type": "video",
            "video": frames,
            "fps": frame_rate,
        },
        {
            "type": "text",
            "text": formatted_qa,
        }
    ]
    
    # The assistant's answer should just be the letter
    assistant_content = example['qa_pairs']['answer']
    
    ret = {
        "messages": [
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ],
        "task_name": "cosmos-r1-reason",
    }
    return ret

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
        
        os.makedirs(video_root_dir, exist_ok=True)
        
        self.formatted_ds = prepare_cosmos_r1_reason_dataset(split=split, task_name=task_name, video_root_dir=video_root_dir)
        self.task_spec = TaskDataSpec(
            task_name="Cosmos R1 Reason",
            prompt_file=prompt_file,
        )

if __name__ == "__main__":
    ds = CosmosR1ReasonDataset(split="default", video_root_dir="/data/cosmos_videos")
