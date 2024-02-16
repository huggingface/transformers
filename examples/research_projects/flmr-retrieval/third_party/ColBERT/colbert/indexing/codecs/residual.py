"""
EVENTUALLY: Tune the batch sizes selected here for a good balance of speed and generality.
"""

import os
import torch
import numpy as np
from itertools import product

from colbert.infra.config import ColBERTConfig
from colbert.indexing.codecs.residual_embeddings import ResidualEmbeddings
from colbert.utils.utils import print_message

import pathlib
from torch.utils.cpp_extension import load


class ResidualCodec:
    Embeddings = ResidualEmbeddings

    def __init__(self, config, centroids, avg_residual=None, bucket_cutoffs=None, bucket_weights=None):
        self.use_gpu = config.total_visible_gpus > 0

        ResidualCodec.try_load_torch_extensions(self.use_gpu)

        if self.use_gpu > 0:
            self.centroids = centroids.cuda().half()
        else:
            self.centroids = centroids.float()
        self.dim, self.nbits = config.dim, config.nbits
        self.avg_residual = avg_residual

        if torch.is_tensor(self.avg_residual):
            if self.use_gpu:
                self.avg_residual = self.avg_residual.cuda().half()

        if torch.is_tensor(bucket_cutoffs):
            if self.use_gpu:
                bucket_cutoffs = bucket_cutoffs.cuda()
                bucket_weights = bucket_weights.half().cuda()

        self.bucket_cutoffs = bucket_cutoffs
        self.bucket_weights = bucket_weights
        if not self.use_gpu and self.bucket_weights is not None:
            self.bucket_weights = self.bucket_weights.to(torch.float32)

        self.arange_bits = torch.arange(0, self.nbits, device='cuda' if self.use_gpu else 'cpu', dtype=torch.uint8)

        self.rank = config.rank

        # We reverse the residual bits because arange_bits as
        # currently constructed produces results with the reverse
        # of the expected endianness
        self.reversed_bit_map = []
        mask = (1 << self.nbits) - 1
        for i in range(256):
            # The reversed byte
            z = 0
            for j in range(8, 0, -self.nbits):
                # Extract a subsequence of length n bits
                x = (i >> (j - self.nbits)) & mask

                # Reverse the endianness of each bit subsequence (e.g. 10 -> 01)
                y = 0
                for k in range(self.nbits - 1, -1, -1):
                    y += ((x >> (self.nbits - k - 1)) & 1) * (2 ** k)

                # Set the corresponding bits in the output byte
                z |= y
                if j > self.nbits:
                    z <<= self.nbits
            self.reversed_bit_map.append(z)
        self.reversed_bit_map = torch.tensor(self.reversed_bit_map).to(torch.uint8)

        # A table of all possible lookup orders into bucket_weights
        # given n bits per lookup
        keys_per_byte = 8 // self.nbits
        if self.bucket_weights is not None:
            self.decompression_lookup_table = (
                torch.tensor(
                    list(
                        product(
                            list(range(len(self.bucket_weights))),
                            repeat=keys_per_byte
                        )
                    )
                )
                .to(torch.uint8)
            )
        else:
            self.decompression_lookup_table = None
        if self.use_gpu:
            self.reversed_bit_map = self.reversed_bit_map.cuda()
            if self.decompression_lookup_table is not None:
                self.decompression_lookup_table = self.decompression_lookup_table.cuda()

    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or not use_gpu:
            return

        print_message(f"Loading decompress_residuals_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...")
        decompress_residuals_cpp = load(
            name="decompress_residuals_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "decompress_residuals.cpp"
                ),
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "decompress_residuals.cu"
                ),
            ],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.decompress_residuals = decompress_residuals_cpp.decompress_residuals_cpp

        print_message(f"Loading packbits_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...")
        packbits_cpp = load(
            name="packbits_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "packbits.cpp"
                ),
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "packbits.cu"
                ),
            ],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.packbits = packbits_cpp.packbits_cpp

        cls.loaded_extensions = True

    @classmethod
    def load(cls, index_path, disable_gpu=False):
        config = ColBERTConfig.load_from_index(index_path)
        if disable_gpu:
            config.total_visible_gpus = 0
        centroids_path = os.path.join(index_path, 'centroids.pt')
        avgresidual_path = os.path.join(index_path, 'avg_residual.pt')
        buckets_path = os.path.join(index_path, 'buckets.pt')

        centroids = torch.load(centroids_path, map_location='cpu')
        avg_residual = torch.load(avgresidual_path, map_location='cpu')
        bucket_cutoffs, bucket_weights = torch.load(buckets_path, map_location='cpu')

        if avg_residual.dim() == 0:
            avg_residual = avg_residual.item()

        return cls(config=config, centroids=centroids, avg_residual=avg_residual, bucket_cutoffs=bucket_cutoffs, bucket_weights=bucket_weights)

    def save(self, index_path):
        assert self.avg_residual is not None
        assert torch.is_tensor(self.bucket_cutoffs), self.bucket_cutoffs
        assert torch.is_tensor(self.bucket_weights), self.bucket_weights

        centroids_path = os.path.join(index_path, 'centroids.pt')
        avgresidual_path = os.path.join(index_path, 'avg_residual.pt')
        buckets_path = os.path.join(index_path, 'buckets.pt')

        torch.save(self.centroids.half(), centroids_path)
        torch.save((self.bucket_cutoffs, self.bucket_weights), buckets_path)

        if torch.is_tensor(self.avg_residual):
            torch.save(self.avg_residual, avgresidual_path)
        else:
            torch.save(torch.tensor([self.avg_residual]), avgresidual_path)

    def compress(self, embs):
        codes, residuals = [], []

        for batch in embs.split(1 << 18):
            if self.use_gpu:
                batch = batch.cuda().half()
            codes_ = self.compress_into_codes(batch, out_device=batch.device)
            centroids_ = self.lookup_centroids(codes_, out_device=batch.device)

            residuals_ = (batch - centroids_)

            codes.append(codes_.cpu())
            residuals.append(self.binarize(residuals_).cpu())

        codes = torch.cat(codes)
        residuals = torch.cat(residuals)

        return ResidualCodec.Embeddings(codes, residuals)

    def binarize(self, residuals):
        residuals = torch.bucketize(residuals.float(), self.bucket_cutoffs).to(dtype=torch.uint8)
        residuals = residuals.unsqueeze(-1).expand(*residuals.size(), self.nbits)  # add a new nbits-wide dim
        residuals = residuals >> self.arange_bits  # divide by 2^bit for each bit position
        residuals = residuals & 1  # apply mod 2 to binarize

        assert self.dim % 8 == 0
        assert self.dim % (self.nbits * 8) == 0, (self.dim, self.nbits)

        if self.use_gpu:
            residuals_packed = ResidualCodec.packbits(residuals.contiguous().flatten())
        else:
            residuals_packed = np.packbits(np.asarray(residuals.contiguous().flatten()))
        residuals_packed = torch.as_tensor(residuals_packed, dtype=torch.uint8)
        residuals_packed = residuals_packed.reshape(residuals.size(0), self.dim // 8 * self.nbits)

        return residuals_packed

    def compress_into_codes(self, embs, out_device):
        """
            EVENTUALLY: Fusing the kernels or otherwise avoiding materalizing the entire matrix before max(dim=0)
                        seems like it would help here a lot.
        """

        codes = []

        bsize = (1 << 29) // self.centroids.size(0)
        for batch in embs.split(bsize):
            if self.use_gpu:
                indices = (self.centroids @ batch.T.cuda().half()).max(dim=0).indices.to(device=out_device)
            else:
                indices = (self.centroids @ batch.T.cpu().float()).max(dim=0).indices.to(device=out_device)
            codes.append(indices)

        return torch.cat(codes)

    def lookup_centroids(self, codes, out_device):
        """
            Handles multi-dimensional codes too.

            EVENTUALLY: The .split() below should happen on a flat view.
        """

        centroids = []

        for batch in codes.split(1 << 20):
            if self.use_gpu:
                centroids.append(self.centroids[batch.cuda().long()].to(device=out_device))
            else:
                centroids.append(self.centroids[batch.long()].to(device=out_device))

        return torch.cat(centroids)

    #@profile
    def decompress(self, compressed_embs: Embeddings):
        """
            We batch below even if the target device is CUDA to avoid large temporary buffers causing OOM.
        """

        codes, residuals = compressed_embs.codes, compressed_embs.residuals

        D = []
        for codes_, residuals_ in zip(codes.split(1 << 15), residuals.split(1 << 15)):
            if self.use_gpu:
                codes_, residuals_ = codes_.cuda(), residuals_.cuda()
                centroids_ = ResidualCodec.decompress_residuals(
                    residuals_,
                    self.bucket_weights,
                    self.reversed_bit_map,
                    self.decompression_lookup_table,
                    codes_,
                    self.centroids,
                    self.dim,
                    self.nbits,
                ).cuda()
            else:
                # TODO: Remove dead code
                centroids_ = self.lookup_centroids(codes_, out_device='cpu')
                residuals_ = self.reversed_bit_map[residuals_.long()]
                residuals_ = self.decompression_lookup_table[residuals_.long()]
                residuals_ = residuals_.reshape(residuals_.shape[0], -1)
                residuals_ = self.bucket_weights[residuals_.long()]
                centroids_.add_(residuals_)

            if self.use_gpu:
                D_ = torch.nn.functional.normalize(centroids_, p=2, dim=-1).half()
            else:
                D_ = torch.nn.functional.normalize(centroids_.to(torch.float32), p=2, dim=-1)
            D.append(D_)

        return torch.cat(D)
