import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from ..utils.generic import ModelOutput


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

        ```
        for item in loader:
            yield infer(item, **params)
        ```

                Arguments:
                    loader (`torch.utils.data.DataLoader` or `Iterable`):
                        The iterator that will be used to apply `infer` on.
                    infer (any function):
                        The function to apply of each element of `loader`.
                    params (`dict`):
                        The parameters passed to `infer` along with every item
                    loader_batch_size (`int`, *optional*):
                        If specified, the items of `loader` are supposed to come as batch, and are loader_batched here
                        making it roughly behave as


        ```
        for items in loader:
            for i in loader_batch_size:
                item = items[i]
                yield infer(item, **params)
        ```"""
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
        """
        Return item located at `loader_batch_index` within the current `loader_batch_data`.
        """
        if isinstance(self._loader_batch_data, torch.Tensor):
            # Batch data is simple tensor, just fetch the slice
            result = self._loader_batch_data[self._loader_batch_index].unsqueeze(0)
        else:
            # Batch data is assumed to be BaseModelOutput (or dict)
            loader_batched = {}
            for k, element in self._loader_batch_data.items():
                if isinstance(element, ModelOutput):
                    # Convert ModelOutput to tuple first
                    element = element.to_tuple()
                    if isinstance(element[0], torch.Tensor):
                        loader_batched[k] = tuple(el[self._loader_batch_index].unsqueeze(0) for el in element)
                    elif isinstance(element[0], np.ndarray):
                        loader_batched[k] = tuple(np.expand_dims(el[self._loader_batch_index], 0) for el in element)
                    continue
                if k in {"hidden_states", "past_key_values", "attentions"} and isinstance(element, tuple):
                    # Those are stored as lists of tensors so need specific unbatching.
                    if isinstance(element[0], torch.Tensor):
                        loader_batched[k] = tuple(el[self._loader_batch_index].unsqueeze(0) for el in element)
                    elif isinstance(element[0], np.ndarray):
                        loader_batched[k] = tuple(np.expand_dims(el[self._loader_batch_index], 0) for el in element)
                    continue
                if element is None:
                    # This can happen for optional data that get passed around
                    loader_batched[k] = None
                elif isinstance(element[self._loader_batch_index], torch.Tensor):
                    # Take correct batch data, but make it looked like batch_size=1
                    # For compatibility with other methods within transformers

                    loader_batched[k] = element[self._loader_batch_index].unsqueeze(0)
                elif isinstance(element[self._loader_batch_index], np.ndarray):
                    # Take correct batch data, but make it looked like batch_size=1
                    # For compatibility with other methods within transformers
                    loader_batched[k] = np.expand_dims(element[self._loader_batch_index], 0)
                else:
                    # This is typically a list, so no need to `unsqueeze`.
                    loader_batched[k] = element[self._loader_batch_index]
            # Recreate the element by reusing the original class to make it look
            # batch_size=1
            result = self._loader_batch_data.__class__(loader_batched)
        self._loader_batch_index += 1
        return result

    def __next__(self):
        if self._loader_batch_index is not None and self._loader_batch_index < self.loader_batch_size:
            # We are currently unrolling a batch so we just need to return
            # the current item within a batch
            return self.loader_batch_item()

        # We're out of items within a batch
        item = next(self.iterator)
        processed = self.infer(item, **self.params)
        # We now have a batch of "inferred things".
        if self.loader_batch_size is not None:
            # Try to infer the size of the batch
            if isinstance(processed, torch.Tensor):
                first_tensor = processed
            elif isinstance(processed, tuple):
                first_tensor = processed[0]
            else:
                key = list(processed.keys())[0]
                first_tensor = processed[key]

            if isinstance(first_tensor, list):
                observed_batch_size = len(first_tensor)
            else:
                observed_batch_size = first_tensor.shape[0]
            if 0 < observed_batch_size < self.loader_batch_size:
                # could be last batch so we can't unroll as many
                # elements.
                self.loader_batch_size = observed_batch_size
            # Setting internal index to unwrap the batch
            self._loader_batch_data = processed[0] if isinstance(processed, tuple) else processed
            self._loader_batch_index = 0
            return self.loader_batch_item()
        else:
            # We're not unrolling batches
            return processed


class PipelineChunkIterator(PipelineIterator):
    def __init__(self, loader, infer, params, loader_batch_size=None):
        """
        Roughly equivalent to

        ```
        for iterator in loader:
            for item in iterator:
                yield infer(item, **params)
        ```

                Arguments:
                    loader (`torch.utils.data.DataLoader` or `Iterable`):
                        The iterator that will be used to apply `infer` on.
                    infer (any function):
                        The function to apply of each element of `loader`.
                    params (`dict`):
                        The parameters passed to `infer` along with every item
        """
        super().__init__(loader, infer, params)

    def __iter__(self):
        self.iterator = iter(self.loader)
        self.subiterator = None
        return self

    def __next__(self):
        if self.subiterator is None:
            "Subiterator None means we haven't started a `preprocess` iterator. so start it"
            self.subiterator = self.infer(next(self.iterator), **self.params)
        try:
            # Try to return next item
            processed = next(self.subiterator)
        except StopIteration:
            # When a preprocess iterator ends, we can start lookig at the next item
            # ChunkIterator will keep feeding until ALL elements of iterator
            # all have created their subiterator and have been iterating against.
            #
            # Another way to look at it, is we're basically flattening lists of lists
            # into a single list, but with generators
            self.subiterator = self.infer(next(self.iterator), **self.params)
            processed = next(self.subiterator)
        return processed


class PipelinePackIterator(PipelineIterator):
    """
    Roughly equivalent to

    ```
    packed =  []
    for item in loader:
        packed.append(item)
        if item["is_last"]:
            yield packed
            packed = []
    ```

        but it also handles cases where `item` are batched (meaning it's a dict of Tensor with first dimension > 1. In
        that case it does

    ```
    packed =  []
    for batch in loader:
        # item is batched
        for item in batch:
            packed.append(item)
            if item["is_last"]:
                yield packed
                packed = []
    ```

        Arguments:
            loader (`torch.utils.data.DataLoader` or `Iterable`):
                The iterator that will be used to apply `infer` on.
            infer (any function):
                The function to apply of each element of `loader`.
            params (`dict`):
                The parameters passed to `infer` along with every item
            loader_batch_size (`int`, *optional*):
                If specified, the items of `loader` are supposed to come as batch, and are loader_batched here making
                it roughly behave as


    ```
    for items in loader:
        for i in loader_batch_size:
            item = items[i]
            yield infer(item, **params)
    ```"""

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __next__(self):
        # Extremely similar to PipelineIterator in its unpacking mechanism
        # BUT, we have an extra required item which is the presence of `is_last`
        # That is because everything is flattened by `PipelineChunkIterator` we
        # need to keep track of how to regroup here in the original `process`
        # boundaries so that `process` and `postprocess` see the same data.

        # This iterator accumulates items (possibly while unbatching) until it
        # its a `is_last` and then just passes it on to the caller.
        is_last = False
        accumulator = []
        if self._loader_batch_index is not None and self._loader_batch_index < self.loader_batch_size:
            while self._loader_batch_index < self.loader_batch_size:
                item = self.loader_batch_item()
                is_last = item.pop("is_last")
                accumulator.append(item)
                if is_last:
                    return accumulator

        while not is_last:
            processed = self.infer(next(self.iterator), **self.params)
            if self.loader_batch_size is not None:
                if isinstance(processed, torch.Tensor):
                    first_tensor = processed
                else:
                    key = list(processed.keys())[0]
                    first_tensor = processed[key]
                if isinstance(first_tensor, list):
                    observed_batch_size = len(first_tensor)
                else:
                    observed_batch_size = first_tensor.shape[0]
                if 0 < observed_batch_size < self.loader_batch_size:
                    # could be last batch so we can't unroll as many
                    # elements.
                    self.loader_batch_size = observed_batch_size
                self._loader_batch_data = processed
                self._loader_batch_index = 0
                while self._loader_batch_index < self.loader_batch_size:
                    item = self.loader_batch_item()
                    is_last = item.pop("is_last")
                    accumulator.append(item)
                    if is_last:
                        return accumulator
            else:
                item = processed
                is_last = item.pop("is_last")
                accumulator.append(item)
        return accumulator


class KeyDataset(Dataset):
    def __init__(self, dataset: Dataset, key: str):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i][self.key]


class KeyPairDataset(Dataset):
    def __init__(self, dataset: Dataset, key1: str, key2: str):
        self.dataset = dataset
        self.key1 = key1
        self.key2 = key2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return {"text": self.dataset[i][self.key1], "text_pair": self.dataset[i][self.key2]}
