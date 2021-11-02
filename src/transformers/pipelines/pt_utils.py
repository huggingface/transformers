import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from transformers.pipelines.audio_utils import ffmpeg_stream, frame_generator, vad_collector


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
        if isinstance(self._loader_batch_data, torch.Tensor):
            result = self._loader_batch_data[self._loader_batch_index]
        else:
            loader_batched = {}
            for k, element in self._loader_batch_data.items():
                if k == "past_key_values":
                    continue
                if isinstance(element[self._loader_batch_index], torch.Tensor):
                    loader_batched[k] = element[self._loader_batch_index].unsqueeze(0)
                elif isinstance(element[self._loader_batch_index], np.ndarray):
                    loader_batched[k] = np.expand_dims(element[self._loader_batch_index], 0)
                else:
                    loader_batched[k] = element[self._loader_batch_index]
            result = self._loader_batch_data.__class__(loader_batched)
        self._loader_batch_index += 1
        return result

    def __next__(self):
        if self._loader_batch_index is not None and self._loader_batch_index < self.loader_batch_size:
            return self.loader_batch_item()

        item = next(self.iterator)
        processed = self.infer(item, **self.params)
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
            return self.loader_batch_item()
        else:
            return processed


class PipelineChunkIterator(PipelineIterator):
    def __iter__(self):
        self.iterator = iter(self.loader)
        self.subiterator = None
        return self

    def __next__(self):
        if self.subiterator is None:
            self.subiterator = self.infer(next(self.iterator), **self.params)
        try:
            processed = next(self.subiterator)
        except StopIteration:
            self.subiterator = self.infer(next(self.iterator), **self.params)
            processed = next(self.subiterator)
        return processed


class PipelinePackIterator(PipelineIterator):
    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __next__(self):
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


class AudioVadDataset(IterableDataset):
    def __init__(self, filenames, sampling_rate: int):
        self.filenames = filenames
        self.sampling_rate = sampling_rate

    def __iter__(self):
        sampling_rate = self.sampling_rate
        try:
            import webrtcvad
        except ImportError:
            raise ValueError(
                "webrtcvad was not found but is required to chunk on voice activation, `pip install webrtcvad`."
            )

        for filename in self.filenames:
            if not isinstance(filename, str):
                raise ValueError("Chunk voice can only operate on large filenames")

            inputs = ffmpeg_stream(filename, self.sampling_rate, format_for_conversion="s16le")
            vad = webrtcvad.Vad(1)
            frames = frame_generator(10, inputs, sampling_rate)
            segments = vad_collector(sampling_rate, 10, 300, vad, frames)
            max_int16 = 2 ** 15
            max_chunk_duration = 20
            max_len = int(max_chunk_duration * sampling_rate)
            for i, segment in enumerate(segments):
                audio = np.frombuffer(segment, dtype=np.int16).astype("float32") / max_int16
                for i in range(0, audio.shape[0], max_len):
                    yield audio[i : i + max_len]


def AudioChunkDataset(IterableDataset):
    def __init__(self, filenames, sampling_rate: int):
        try:
            from scipy import signal
        except ImportError:
            raise ValueError("scipy was not found but is required to chunk on voice activation, `pip install scipy`.")

        self.filenames = filenames
        self.sampling_rate = sampling_rate

        f1 = 50  # 50Hz
        f2 = 300  # 300Hz
        fs = sampling_rate

        nyq = 0.5 * fs
        low = f1 / nyq
        high = f2 / nyq
        order = 10
        self.sos = signal.butter(order, [low, high], analog=False, btype="band", output="sos")
        chunk_min_duration = 5
        chunk_max_duration = 20
        chunk_pad_duration = 0.3
        self.start_chunk = int(sampling_rate * chunk_min_duration)
        self.stop_chunk = int(sampling_rate * chunk_max_duration)
        self.pad_chunk = int(sampling_rate * chunk_pad_duration)

    def __iter__(self):
        from scipy import signal

        for filename in self.filenames:
            leftover = np.zeros((0,), dtype=np.float32)
            pad = np.zeros((self.pad_chunk,), dtype=np.float32)
            if not isinstance(filename, str):
                raise ValueError("Chunk voice can only operate on large filenames")

            for audio in ffmpeg_stream(filename, self.sampling_rate):
                audio = np.concatenate([leftover, audio])
                chunk_portion = audio[self.start_chunk : self.stop_chunk]
                if chunk_portion.shape[0] == 0:
                    padded = np.concatenate([pad, audio, pad])
                    yield padded
                    break
                voice_filtered = signal.sosfilt(self.sos, chunk_portion)
                index = self.start_chunk + voice_filtered.argmin()
                chunked = audio[:index]

                leftover = audio[index:]

                padded = np.concatenate([pad, chunked, pad])
                yield padded
