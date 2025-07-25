from transformers import PreTrainedTokenizerBase
from typing import Union
import torch
import numpy as np

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
    
    def as_tensor(self, as_tensors: bool = True) -> torch.Tensor:
        return self.tensor if as_tensors else self
    
    def __len__(self) -> int:
        # this is the number of items in the batch
        return len(self.num_elements_per_item)
    
    def to(self, device: str | torch.device) -> "PackedMultimodalDataBatch":
        self.tensor = self.tensor.to(device)

    def slice(self, indices: Union[list[int], torch.Tensor]) -> "PackedMultimodalDataBatch":
        idx = indices.tolist() if isinstance(indices, torch.Tensor) else indices
        # cumulative sum of the number of elements per item
        cu_sum = np.cumsum([0] + self.num_elements_per_item)
        start_idx = cu_sum[:-1]
        end_idx = cu_sum[1:]
        # if contiguous list of indices, we can use index_select from start of first to end of last
        selected_indices = _create_indices_from_list(start_idx, end_idx)
        tensor = self.tensor.index_select(self.dim_to_pack, torch.IntTensor(selected_indices))
        sliced_num_elements_per_item = [end_idx[i] - start_idx[i] for i in idx]
        return PackedMultimodalDataBatch(tensor, self.dim_to_pack, sliced_num_elements_per_item)

    
    def as_packed_multimodal_data_item(self) -> PackedMultimodalDataItem:
        # you can collapse the batch into a single item by calling this function
        # called inside `message_log_to_flat_messages` to convert the multimodal tensors from turns into a conversation
        return PackedMultimodalDataItem(self.tensor, self.dim_to_pack)
    
    def repeat_interleave(self, num_repeats: int) -> "PackedMultimodalDataBatch":
        """Repeats the batch num_repeats times."""
        raise NotImplementedError("Not implemented")


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
    all_keys.update(processor.model_input_names)
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



    