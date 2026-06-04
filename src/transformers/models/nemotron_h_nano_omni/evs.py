import torch


class EfficientVideoSampling:
    @staticmethod
    def compute_retention_mask(
        *,
        video_embeds: torch.FloatTensor,
        thw: torch.LongTensor,
        spatial_merge_size: int,
        q: float,
    ):
        """
        Computes the retention mask for video embeddings based on the grid dimensions.

        Args:
            video_embeds (`torch.FloatTensor` of shape `(T * H * W, hidden_size)`):
                The video embeddings to compute the retention mask for.
            thw (`torch.LongTensor` of shape `(3)`):
                The temporal, height and width of feature shape of each video in LLM.
            spatial_merge_size (`int`): The spatial merge size of the video embeddings.
                If embeddings will be downsampled *later*, this should be the downsampling factor.
            q: (`float`): Pruning rate factor, indicating number of tokens to prune (remove)

        Returns:
            `torch.Tensor`: The retention mask for the video embeddings (T * H * W).
                1 for tokens to keep, 0 for tokens to prune.
        """
        T, H, W = thw

        # video_embeds = einops.rearrange(
        #     video_embeds,
        #     "(T H W) C -> T H W C",
        #     T=T,
        #     H=H // spatial_merge_size,
        #     W=W // spatial_merge_size,
        # )
        # Use reshape instead of einops to avoid graph breaks
        video_embeds = video_embeds.reshape(T, H // spatial_merge_size, W // spatial_merge_size, video_embeds.size(-1))

        # Core EVS
        similarity = torch.nn.functional.cosine_similarity(video_embeds[1:, ...], video_embeds[:-1, ...], dim=-1)
        dissimilarity = 1 - similarity

        # Always ensure we include all tokens from the first frame
        dissimilarity = torch.cat([255 * torch.ones_like(video_embeds[:1, :, :, 0]), dissimilarity], dim=0)
        dissimilarity_flat = dissimilarity.view(-1)

        min_num_tokens = (H // spatial_merge_size) * (W // spatial_merge_size)  # a single frame
        evs_num_tokens = int(T * min_num_tokens * (1 - q))
        num_tokens_to_keep = max(min_num_tokens, evs_num_tokens)

        order = torch.argsort(dissimilarity_flat, dim=-1, descending=True, stable=True)
        topk_indices = order[:num_tokens_to_keep]

        retention_mask = torch.zeros_like(dissimilarity_flat, dtype=torch.bool)
        retention_mask[topk_indices] = True
        retention_mask = retention_mask.reshape(dissimilarity.size())

        # print(
        #     f"Computed retention mask of shape {retention_mask.shape=} with sparsity {retention_mask.float().mean().item():.4f} for {q=}",
        # )
        mask = retention_mask.view(-1)  # "T H W -> (T H W)"
        return mask
