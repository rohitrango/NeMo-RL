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

from typing import Any, Optional

from datasets import Dataset, load_dataset

from nemo_rl.data.interfaces import TaskDataSpec

def format_answer_fromtags(answer: str) -> str:
    """
    Format the answer from tags.
    """
    # Extract content between <answer> tags and strip whitespace
    import re
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, answer)
    return match.group(1).strip() if match else ""

def format_clevr_cogent_dataset(example: dict[str, Any]) -> dict[str, Any]:
    """
    Format the CLEVR-CoGenT dataset.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": example['image'],
                    },
                    {
                        "type": "text",
                        "text": example["problem"],
                    }
                ]
            },
            {
                "role": "assistant",
                "content": format_answer_fromtags(example["solution"]),
            },
        ],
        "task_name": "CLEVR-CoGenT",
    }
    

# contain different variants of the CLEVR dataset 
def prepare_clevr_cogent_dataset(split: str = "trainA", seed: int = 42):

    if split == "trainA":
        tr_dataset = load_dataset("MMInstruction/Clevr_CoGenT_TrainA_70K_Complex")['train']
        val_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValA")['train']
    elif split == "trainB":
        tr_dataset = load_dataset("MMInstruction/Clevr_CoGenT_TrainB_70K_Complex")['train']
        val_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB")['train']
    elif split == "valA":
        tr_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValA")['train']
        val_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValA")['train']
    elif split == "valB":
        tr_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB")['train']
        val_dataset = load_dataset("MMInstruction/Clevr_CoGenT_ValB")['train']
    
    # format
    tr_dataset = tr_dataset.map(format_clevr_cogent_dataset)
    val_dataset = val_dataset.map(format_clevr_cogent_dataset)

    return {
        'train': tr_dataset,
        'validation': val_dataset,
    }
        

class CLEVRCoGenTDataset:
    def __init__(self, split: str = "trainA", seed: int = 42, prompt_file: Optional[str] = None):
        """
        """
        if split not in ['trainA', 'trainB', 'valA', 'valB']:
            raise ValueError(f"Invalid split: {split}. Please use 'trainA', 'trainB', 'valA', or 'valB'.")
        
        # self.formatted_ds = load_dataset("MMInstruction/Clevr_CoGenT_TrainA_70K_Complex", split=split)
        self.formatted_ds = prepare_clevr_cogent_dataset(split=split, seed=seed, )
        self.task_spec = TaskDataSpec(
            task_name="CLEVR",
            prompt_file=prompt_file,
        )
        
