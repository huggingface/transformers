import os
import queue
import ujson
import threading

from contextlib import contextmanager

from colbert.indexing.codecs.residual import ResidualCodec

from colbert.utils.utils import print_message


class IndexSaver():
    def __init__(self, config):
        self.config = config

    def save_codec(self, codec):
        codec.save(index_path=self.config.index_path_)

    def load_codec(self):
        return ResidualCodec.load(index_path=self.config.index_path_)

    def try_load_codec(self):
        try:
            ResidualCodec.load(index_path=self.config.index_path_)
            return True
        except Exception as e:
            return False

    def check_chunk_exists(self, chunk_idx):
        # TODO: Verify that the chunk has the right amount of data?

        doclens_path = os.path.join(self.config.index_path_, f'doclens.{chunk_idx}.json')
        if not os.path.exists(doclens_path):
            return False

        metadata_path = os.path.join(self.config.index_path_, f'{chunk_idx}.metadata.json')
        if not os.path.exists(metadata_path):
            return False

        path_prefix = os.path.join(self.config.index_path_, str(chunk_idx))
        codes_path = f'{path_prefix}.codes.pt'
        if not os.path.exists(codes_path):
            return False

        residuals_path = f'{path_prefix}.residuals.pt'  # f'{path_prefix}.residuals.bn'
        if not os.path.exists(residuals_path):
            return False

        return True

    @contextmanager
    def thread(self):
        self.codec = self.load_codec()

        self.saver_queue = queue.Queue(maxsize=3)
        thread = threading.Thread(target=self._saver_thread)
        thread.start()

        try:
            yield

        finally:
            self.saver_queue.put(None)
            thread.join()

            del self.saver_queue
            del self.codec

    def save_chunk(self, chunk_idx, offset, embs, doclens):
        compressed_embs = self.codec.compress(embs)

        self.saver_queue.put((chunk_idx, offset, compressed_embs, doclens))

    def _saver_thread(self):
        for args in iter(self.saver_queue.get, None):
            self._write_chunk_to_disk(*args)

    def _write_chunk_to_disk(self, chunk_idx, offset, compressed_embs, doclens):
        path_prefix = os.path.join(self.config.index_path_, str(chunk_idx))
        compressed_embs.save(path_prefix)

        doclens_path = os.path.join(self.config.index_path_, f'doclens.{chunk_idx}.json')
        with open(doclens_path, 'w') as output_doclens:
            ujson.dump(doclens, output_doclens)

        metadata_path = os.path.join(self.config.index_path_, f'{chunk_idx}.metadata.json')
        with open(metadata_path, 'w') as output_metadata:
            metadata = {'passage_offset': offset, 'num_passages': len(doclens), 'num_embeddings': len(compressed_embs)}
            ujson.dump(metadata, output_metadata)
