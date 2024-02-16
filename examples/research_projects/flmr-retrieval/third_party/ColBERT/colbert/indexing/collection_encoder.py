import torch

from colbert.infra.run import Run
from colbert.utils.utils import print_message, batch


class CollectionEncoder():
    def __init__(self, config, checkpoint):
        self.config = config
        self.checkpoint = checkpoint
        self.use_gpu = self.config.total_visible_gpus > 0

    def encode_passages(self, passages):
        Run().print(f"#> Encoding {len(passages)} passages..")

        if len(passages) == 0:
            return None, None

        with torch.inference_mode():
            embs, doclens = [], []

            # Batch here to avoid OOM from storing intermediate embeddings on GPU.
            # Storing on the GPU helps with speed of masking, etc.
            # But ideally this batching happens internally inside docFromText.
            for passages_batch in batch(passages, self.config.bsize * 50):
                embs_, doclens_ = self.checkpoint.docFromText(passages_batch, bsize=self.config.bsize,
                                                              keep_dims='flatten', showprogress=(not self.use_gpu))
                embs.append(embs_)
                doclens.extend(doclens_)

            embs = torch.cat(embs)

            # embs, doclens = self.checkpoint.docFromText(passages, bsize=self.config.bsize,
            #                                                   keep_dims='flatten', showprogress=(self.config.rank < 1))

        # with torch.inference_mode():
        #     embs = self.checkpoint.docFromText(passages, bsize=self.config.bsize,
        #                                        keep_dims=False, showprogress=(self.config.rank < 1))
        #     assert type(embs) is list
        #     assert len(embs) == len(passages)

        #     doclens = [d.size(0) for d in embs]
        #     embs = torch.cat(embs)

        return embs, doclens
