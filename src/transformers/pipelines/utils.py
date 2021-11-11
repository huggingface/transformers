from abc import ABC, abstractmethod

import numpy as np

from ..file_utils import is_torch_available
from ..utils import logging


logger = logging.get_logger(__name__)

if is_torch_available():
    from tensorflow import Tensor as TFTensor
else:

    class TFTensor:
        pass


if is_torch_available():
    import torch
    from torch import Tensor as TorchTensor
    from torch.utils.data import DataLoader, Dataset, IterableDataset
else:

    class TorchTensor:
        pass

    class Dataset(ABC):
        @abstractmethod
        def __len__(self, i):
            raise NotImplementedError("Implement __len__")

        @abstractmethod
        def __getitem__(self, i):
            raise NotImplementedError("Implement __getitem__")

    class IterableDataset(ABC):
        @abstractmethod
        def __iter__(self):
            raise NotImplementedError("Implement __iter__")

        @abstractmethod
        def __next__(self):
            raise NotImplementedError("Implement __next__")

    class DataLoader:
        def __init__(self, dataset, num_workers=0, batch_size=1, collate_fn=None):
            self.dataset = dataset
            self.num_workers = num_workers
            if self.num_workers > 0:
                logger.warning("For non pytorch, we use a dummy dataloader that does" " not implement `num_workers`.")
            self.batch_size = batch_size
            self.collate_fn = collate_fn

        def __iter__(self):
            self.iter = iter(self.dataset)
            return self

        def __next__(self):
            if self.iter is None:
                raise StopIteration
            batch = []
            for _ in range(self.batch_size):
                try:
                    item = next(self.iter)
                except StopIteration:
                    if len(batch) == 0:
                        raise
                    else:
                        # Raise on next iteration
                        self.iter = None
                        break
                batch.append(item)
            return self.collate_fn(batch)


class PipelineDataset(Dataset):
    def __init__(self, dataset, process, params):
        self.dataset = dataset
        self.process = process
        self.params = params

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        processed = self.process(item, **self.params)
        return processed


class PipelineIterator(IterableDataset):
    def __init__(self, loader, infer, params, loader_batch_size=None):
        """
        Roughly equivalent to

        .. code-block::
            for item in loader:
                yield infer(item, **params)

        Arguments:
            loader (:obj:`torch.utils.data.DataLoader` or any iterator):
                The iterator that will be used to apply :obj:`infer` on.
            infer (any function):
                The function to apply of each element of :obj:`loader`.
            params (:obj:`dict`):
                The parameters passed to :obj:`infer` along with every item
            loader_batch_size (:obj:`int`, `optional`):
                If specified, the items of :obj:`loader` are supposed to come as batch, and are loader_batched here
                making it roughly behave as


                .. code-block::

                    for items in loader:
                        for i in loader_batch_size:
                            item = items[i]
                            yield infer(item, **params)
        """
        self.loader = loader
        self.infer = infer
        self.params = params
        if loader_batch_size == 1:
            # Let's spare some time by deactivating altogether
            loader_batch_size = None
        self.loader_batch_size = loader_batch_size

        # Internal bookkeeping
        self._loader_batch_index = None
        self._loader_batch_data = None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def loader_batch_item(self):
        if isinstance(self._loader_batch_data, TorchTensor):
            result = self._loader_batch_data[self._loader_batch_index]
        else:
            loader_batched = {}
            for k, element in self._loader_batch_data.items():
                if k == "past_key_values":
                    continue
                if isinstance(element[self._loader_batch_index], TorchTensor):
                    loader_batched[k] = element[self._loader_batch_index].unsqueeze(0)
                elif isinstance(element[self._loader_batch_index], np.ndarray):
                    loader_batched[k] = np.expand_dims(element[self._loader_batch_index], 0)
                else:
                    loader_batched[k] = element[self._loader_batch_index]
            result = self._loader_batch_data.__class__(**loader_batched)
        self._loader_batch_index += 1
        return result

    def __next__(self):
        if self._loader_batch_index is not None and self._loader_batch_index < self.loader_batch_size:
            return self.loader_batch_item()

        item = next(self.iterator)
        processed = self.infer(item, **self.params)
        if self.loader_batch_size is not None:
            if isinstance(processed, TorchTensor):
                first_tensor = processed
            else:
                key = list(processed.keys())[0]
                first_tensor = processed[key]
            if isinstance(first_tensor, list):
                observed_batch_size = len(first_tensor)
            else:
                observed_batch_size = first_tensor.shape[0]
            if 0 < observed_batch_size < self.loader_batch_size:
                # Could be last batch so we can't unroll as many
                # elements.
                self.loader_batch_size = observed_batch_size
            self._loader_batch_data = processed
            self._loader_batch_index = 0
            return self.loader_batch_item()
        else:
            return processed


class KeyDataset(Dataset):
    def __init__(self, dataset: Dataset, key: str):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i][self.key]


def no_collate_fn(items):
    if len(items) != 1:
        raise ValueError("This collate_fn is meant to be used with batch_size=1")
    return items[0]


def _pad(items, key, padding_value, padding_side, align_to=None):
    batch_size = len(items)
    if isinstance(items[0][key], (TorchTensor, np.ndarray)):
        if isinstance(items[0][key], TorchTensor):
            cat = torch.cat
            zeros = torch.zeros
            is_torch = True
        else:
            cat = np.concatenate
            zeros = np.zeros
            is_torch = False

        # Others include `attention_mask` etc...
        shape = items[0][key].shape
        dim = len(shape)
        if dim == 4:
            # This is probable image so padding shouldn't be necessary
            # B, C, H, W
            return cat([item[key] for item in items], dim=0)
        max_length = max(item[key].shape[1] for item in items)
        if align_to is not None:
            r = max_length % align_to
            if r != 0:
                max_length += align_to - r

        dtype = items[0][key].dtype

        if dim == 2:
            tensor = zeros((batch_size, max_length), dtype=dtype) + padding_value
        elif dim == 3:
            tensor = zeros((batch_size, max_length, shape[-1]), dtype=dtype) + padding_value

        for i, item in enumerate(items):
            small_tensor = item[key][0]
            if is_torch:
                tensor = tensor.clone()
            if dim == 2:
                if padding_side == "left":
                    tensor[i, -len(small_tensor) :] = small_tensor
                else:
                    tensor[i, : len(small_tensor)] = small_tensor
            elif dim == 3:
                if padding_side == "left":
                    tensor[i, -len(small_tensor) :, :] = small_tensor
                else:
                    tensor[i, : len(small_tensor), :] = small_tensor
        return tensor
    else:
        return [item[key] for item in items]


def pad_collate_fn(tokenizer, feature_extractor, align_to=None):
    padding_side = "right"
    if tokenizer is None and feature_extractor is None:
        raise ValueError("Pipeline without tokenizer or feature_extractor cannot do batching")
    if tokenizer is not None:
        if tokenizer.pad_token_id is None:
            raise ValueError(
                "Pipeline with tokenizer without pad_token cannot do batching. You can try to set it with "
                "`pipe.tokenizer.pad_token_id = model.config.eos_token_id`."
            )
        else:
            padding_value = tokenizer.pad_token_id
            padding_side = tokenizer.padding_side
    if feature_extractor is not None:
        # Feature extractor can be images, where no padding is expected
        padding_value = getattr(feature_extractor, "padding_value", None)
        padding_side = getattr(feature_extractor, "padding_side", None)

    def inner(items):
        keys = set(items[0].keys())
        for item in items:
            if set(item.keys()) != keys:
                raise ValueError(
                    f"The elements of the batch contain different keys. Cannot batch them ({set(item.keys())} != {keys})"
                )
        # input_values, input_pixels, input_ids, ...
        padded = {
            key: _pad(items, key, padding_value if key.startswith("input_") else 0, padding_side, align_to=align_to)
            for key in keys
        }
        return padded

    return inner


def align_multiple_of_8(model_inputs):
    "Flax only feature to take advantage of `jit` by forcing alignment"
