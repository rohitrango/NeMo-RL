# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from torch.nn import functional as F
from typing import Union

from transformers.models.qwen2_vl.modeling_qwen2_vl import PatchEmbed, Qwen2VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

### QwenVL Hotfixes ###
def fix_qwen2vl_patch_embed(model: Union[Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration]):
    def forward_linear_hotfix(self, x: torch.Tensor):
        w_linear = self.proj.weight.flatten(1)
        return F.linear(x, w_linear)
    # replace the conv3d with linear in the PatchEmbed
    model.visual.patch_embed.forward = forward_linear_hotfix.__get__(model.visual.patch_embed, PatchEmbed)
    return model


def apply_dtensor_policy_hotfix(model):
    ### apply hotfixes here ###
    if isinstance(model, (Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration)):
        print("Applying QwenVL hotfix")
        model = fix_qwen2vl_patch_embed(model)
    return model