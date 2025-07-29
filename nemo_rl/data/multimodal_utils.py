from transformers import PreTrainedTokenizerBase
from typing import Union, Optional
import torch
import numpy as np
from collections import defaultdict
import re

def _create_indices_from_list(start_idx: list[int], end_idx: list[int]) -> torch.Tensor:
    '''
    Create a list of indices from a list of start and end indices.
    '''
    indices = []
    for s, e in zip(start_idx, end_idx):
        indices.extend(list(range(s, e)))
    return indices

class PackedMultimodalDataItem:
    """Wrapper around a torch tensor that contains multimodal data"""
    def __init__(self, tensor: torch.Tensor, dim_to_pack: int = 0):
        self.tensor = tensor
        self.dim_to_pack = dim_to_pack
    
    def __call__(self):
        return self.tensor

class PackedMultimodalDataBatch:
    """Wrapper around a torch tensor that contains multimodal data"""
    def __init__(self, tensor: Union[torch.Tensor, list[PackedMultimodalDataItem]], dim_to_pack: int, num_elements_per_item: list[int] = None):
        # this is a concatenated databatch of packed multimodal data
        if isinstance(tensor, torch.Tensor):
            self.tensor = tensor
            self.dim_to_pack = dim_to_pack
            self.num_elements_per_item = num_elements_per_item.tolist() if isinstance(num_elements_per_item, torch.Tensor) else num_elements_per_item
        else:
            raise ValueError("tensor must be a torch.Tensor or a list of PackedMultimodalDataItem objects")
    
    def as_tensor(self, as_tensors: bool = True, device: Optional[torch.device] = None) -> torch.Tensor:
        return (self.tensor.to(device) if device is not None else self.tensor) if as_tensors else self
    
    def __len__(self) -> int:
        # this is the number of items in the batch
        return len(self.num_elements_per_item)
    
    def to(self, device: str | torch.device) -> "PackedMultimodalDataBatch":
        self.tensor = self.tensor.to(device)
        return self

    def slice(self, indices: Union[list[int], torch.Tensor]) -> "PackedMultimodalDataBatch":
        idx = indices.tolist() if isinstance(indices, torch.Tensor) else indices
        # cumulative sum of the number of elements per item
        cu_sum = np.cumsum([0] + self.num_elements_per_item)
        start_idx_all = cu_sum[:-1]
        end_idx_all = cu_sum[1:]
        # select the indices
        start_idx = [start_idx_all[i] for i in idx]
        end_idx = [end_idx_all[i] for i in idx]
        # if contiguous list of indices, we can use index_select from start of first to end of last
        selected_indices = _create_indices_from_list(start_idx, end_idx)
        tensor = self.tensor.index_select(self.dim_to_pack, torch.tensor(selected_indices, device=self.tensor.device).to(torch.int32))
        sliced_num_elements_per_item = [end_idx[i] - start_idx[i] for i in range(len(idx))]
        return PackedMultimodalDataBatch(tensor, self.dim_to_pack, sliced_num_elements_per_item)

    
    def as_packed_multimodal_data_item(self) -> PackedMultimodalDataItem:
        # you can collapse the batch into a single item by calling this function
        # called inside `message_log_to_flat_messages` to convert the multimodal tensors from turns into a conversation
        return PackedMultimodalDataItem(self.tensor, self.dim_to_pack)
    
    def repeat_interleave(self, num_repeats: int) -> "PackedMultimodalDataBatch":
        """Repeats the batch num_repeats times."""
        raise NotImplementedError("Why are you interleaving batches? Interleaving is supposed to happen at the message log level. \
                                  If you are adding new functionality, implement this function.")


def get_multimodal_keys_from_processor(processor) -> list[str]:
    '''
    Get keys of the multimodal data that can be used as model inputs.

    This will be used in the data_processor function to determine which keys to use as model inputs.
    '''
    if isinstance(processor, PreTrainedTokenizerBase):
        return []
    
    all_keys = set()
    if hasattr(processor, "image_processor"):
        all_keys.update(processor.image_processor.model_input_names)
    if hasattr(processor, "video_processor"):
        all_keys.update(processor.video_processor.model_input_names)
    if hasattr(processor, "feature_extractor"):
        all_keys.update(processor.feature_extractor.model_input_names)
    # all_keys.update(processor.model_input_names)
    all_keys.difference_update(set(processor.tokenizer.model_input_names))
    return list(all_keys)

def concat_packed_multimodal_items(packed_multimodal_data: list[PackedMultimodalDataItem], return_as_item: bool = False) -> Union[PackedMultimodalDataBatch, PackedMultimodalDataItem]:
    """Concatenate a list of PackedMultimodalDataItem objects into a single PackedMultimodalDataBatch.
    """
    dim = [item.dim_to_pack for item in packed_multimodal_data]
    assert len(set(dim)) == 1, "All packed multimodal data must have the same dim_to_pack"
    dim = dim[0]
    tensor = torch.cat([item.tensor for item in packed_multimodal_data], dim=dim)

    num_items = [item.tensor.shape[dim] for item in packed_multimodal_data]
    # packed batch
    batch = PackedMultimodalDataBatch(tensor, dim, num_items)
    if return_as_item:
        return batch.as_packed_multimodal_data_item()
    return batch

def concat_packed_multimodal_batches(packed_batches: list[PackedMultimodalDataBatch]) -> PackedMultimodalDataBatch:
    """Concatenate a list of PackedMultimodalDataBatch objects into a single PackedMultimodalDataBatch.

    Each batch must have the same dim_to_pack.
    """
    dim_to_packs = [batch.dim_to_pack for batch in packed_batches]
    assert len(set(dim_to_packs)) == 1, "All packed multimodal data must have the same dim_to_pack"
    # concatenate the tensors
    tensors = [batch.tensor for batch in packed_batches]
    dim_to_pack = dim_to_packs[0]
    tensor = torch.cat(tensors, dim=dim_to_pack)
    num_items = []
    for batch in packed_batches:
        num_items.extend(batch.num_elements_per_item)
    return PackedMultimodalDataBatch(tensor, dim_to_pack, num_items)


def reroute_processor_model_name_patch(model_name: str) -> str:
    '''
    for certain models, the processor is configured incorrectly, so we use another processor that is safer

    First, we try to match the model name to an exact match in the registry.
    If not found, we try to match the model name to a regex in the registry.
    '''
    PROCESSOR_REROUTE_REGEX_REGISTRY = {}
    PROCESSOR_REROUTE_EXACT_REGISTRY = {}
    regex_registry = PROCESSOR_REROUTE_REGEX_REGISTRY

    # Qwen2-VL models have a processor that produces empty strings in `apply_chat_template` function
    regex_registry.update({'qwen/qwen2-vl-*': 'Qwen/Qwen2.5-VL-3B-Instruct'})

    if model_name.lower() in PROCESSOR_REROUTE_EXACT_REGISTRY:
        print(f"Rerouting processor for {model_name} to {PROCESSOR_REROUTE_EXACT_REGISTRY[model_name.lower()]}")
        return PROCESSOR_REROUTE_EXACT_REGISTRY[model_name.lower()]

    for regex, replacement in regex_registry.items():
        if re.match(regex, model_name.lower()):
            print(f"Rerouting processor for {model_name} to {replacement}")
            return replacement
    return model_name

def augment_processor_with_chat_template(processor, model_name: str):
    ''' given a processor, augment it with a chat template

    there have to be two implementations - 
    1) one with tokenize=True, 
    2) one with tokenize=False
    '''
    # return processor
    if 'phi-4' in model_name.lower():
        def conversation_preprocessor_phi4(self, conversation):
            # get new conversation with tokenizer
            new_conversation = []
            image_id = 1
            audio_id = 1
            for turn in conversation:
                new_turn = {}
                new_turn['role'] = turn['role']
                new_turn['content'] = ""
                # if just text, copy it, else create a text string with placeholders
                if isinstance(turn['content'], str):
                    new_turn['content'] = turn['content']
                elif isinstance(turn['content'], list):
                    for item in turn['content']:
                        if item['type'] == 'text':
                            new_turn['content'] += item['text']
                        elif item['type'] == 'image':
                            new_turn['content'] += f"<|image_{image_id}|>"
                            image_id += 1
                        elif item['type'] == 'audio':
                            new_turn['content'] += f"<|audio_{audio_id}|>"
                            audio_id += 1
                        else:
                            raise ValueError(f"Unexpected content type: {item['type']}")
                # else:
                    # raise ValueError(f"Unexpected content type: {type(turn['content'])}")
                new_conversation.append(new_turn)
            # call the original apply_chat_template
            return new_conversation
        print(f"Augmenting processor for {model_name} with phi-4 chat template conversation_preprocessor")
        processor.conversation_preprocessor = conversation_preprocessor_phi4

    return processor
        
def get_dim_to_pack_along(processor, key: str) -> int:
    '''
    Special considerations for packing certain keys from certain processors

    In most cases, the packed items are along dim 0
    '''
    if processor.__class__.__name__ == "SmolVLMProcessor":
        return 1
    # return zero by default
    return 0
