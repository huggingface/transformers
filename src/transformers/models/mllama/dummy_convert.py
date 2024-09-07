import torch


PRENAME_MAPPING = {
    "text_model.layers.": "language_model.layers",
    "attention.wqkv.layer_norm_weight":"input_layernorm.weight",
    "feed_forward.mlp,.layer_norm_weight":"post_attention_layernorm.weight",
    "attention.wo.weight":"self_attn.o_proj.weight",
    "feed_forward.mlp.fc2_weight":"mlp.down_proj.weigh",
    # language model specifics
    "text_model.norm.weight":"model.language_model.norm.weigh",
    "text_model.tok_embeddings.weight":"model.language_model.embed_tokens.weight",
    "text_model.learnable_embedding.weight":"model.language_model.learnable_embedding.weight",
    "text_model.output.weight":"lm_head.weight",
    # cross attention layers
    "cross_attention_layers":"layers", # pretty sure that this is the easiest for our codebase
}





class GatedPositionalEmbeddingModel:
    def __init__(self, aspect_ratios, gated_positional_embedding, embedding_gate):
        self.aspect_ratios = aspect_ratios
        self.gated_positional_embedding = gated_positional_embedding
        self.positional_embedding = torch.nn.Parameter(torch.randn(30, 60))
        self.gated_positional_embedding_gate = embedding_gate
        self.precomputed_embeddings = None
        self.mask = None  # Mask to track valid tiles
        self.precompute_positional_embeddings_and_mask()

    def precompute_positional_embeddings_and_mask(self):
        # Precompute and stack all positional embeddings and mask
        embeddings = []
        mask_list = []
        max_tiles = 0  # Track max number of tiles for padding

        for aspect_ratio_h, aspect_ratio_w in self.aspect_ratios:
            num_tiles = aspect_ratio_h * aspect_ratio_w
            embedding = self.gated_positional_embedding[:aspect_ratio_h, :aspect_ratio_w]
            reshaped_embedding = embedding.reshape(num_tiles, -1)  # Flatten height/width

            embeddings.append(reshaped_embedding)
            mask_list.append(torch.ones(num_tiles, dtype=torch.bool))  # Mask for valid tiles

            max_tiles = max(max_tiles, num_tiles)

        # Pad embeddings and masks so all have the same number of tiles
        self.precomputed_embeddings = torch.stack([
            torch.nn.functional.pad(embed * self.gated_positional_embedding_gate.tanh(), (0, 0, 0, max_tiles - embed.shape[0]))
            for embed in embeddings
        ])

        # Create a padded mask for each entry (pad with zeros for invalid tiles)
        self.mask = torch.stack([
            torch.nn.functional.pad(m, (0, max_tiles - m.shape[0]), value=0)
            for m in mask_list
        ])

    def apply_gated_positional_embedding(self, hidden_state):
        # Apply the gate to all precomputed embeddings at once
        gate = 1 - self.gated_positional_embedding_gate.tanh()
        gated_positional_embeddings_with_gate = gate * self.positional_embedding
        # Expand mask dimensions to match the hidden state shape for broadcasting
        mask_expanded = self.mask.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, max_tiles, 1, 1)
        # Use masked_scatter_ to update hidden_state
        hidden_state.masked_scatter_(mask_expanded, gated_positional_embeddings_with_gate)
        return hidden_state

    def apply_gated_positional_embedding2(self, hidden_state: torch.Tensor, aspect_ratios: torch.Tensor) -> torch.Tensor:

        bsz, num_chunks, num_tokens, dim = hidden_state.shape
        hidden_state = hidden_state.view(bsz * num_chunks, num_tokens, dim)

        # apply regular positional embedding with gate
        gate = 1 - self.gated_positional_embedding_gate.tanh()
        hidden_state = hidden_state + gate * self.positional_embedding

        hidden_state = hidden_state.view(bsz, num_chunks, num_tokens, dim)

        # apply gated positional embedding with gate
        for idx, (aspect_ratio_h, aspect_ratio_w) in enumerate(aspect_ratios):
            num_tiles = aspect_ratio_h * aspect_ratio_w
            gated_positional_embedding = self.gated_positional_embedding[:aspect_ratio_h, :aspect_ratio_w]
            embedding_height, embedding_width = gated_positional_embedding.shape[2:]
            gated_positional_embedding = gated_positional_embedding.reshape(num_tiles, embedding_height, embedding_width)
            gate = self.gated_positional_embedding_gate.tanh()
            gated_positional_embedding_with_gate = gate * gated_positional_embedding
            hidden_state[idx, :num_tiles] += gated_positional_embedding_with_gate

        return hidden_state

# Initialize random seed for reproducibility
torch.manual_seed(42)

# Dummy input parameters for testing
aspect_ratios = [(2, 3), (3, 3), (2, 2)]  # Different aspect ratios
gated_positional_embedding = torch.randn(3, 3, 4, 4)  # Assuming embedding of shape (3, 3, 4, 4) for testing
embedding_gate = torch.randn(1)  # Single gate value

# Initialize hidden state with a specific shape (batch_size, max_tiles, height, width)
batch_size = len(aspect_ratios)
max_tiles = max([h * w for h, w in aspect_ratios])
hidden_state = torch.zeros(batch_size, max_tiles, 4, 4)  # Assuming embeddings are (4x4)

# Initialize the model with precomputed embeddings and mask
model = GatedPositionalEmbeddingModel(aspect_ratios, gated_positional_embedding, embedding_gate)

# Apply the gated positional embeddings to the hidden state
print("Hidden state before applying gated positional embedding:")
print(hidden_state)

out1 = model.apply_gated_positional_embedding(hidden_state)
out2 = model.apply_gated_positional_embedding2(hidden_state, aspect_ratios)
print("\nHidden state after applying gated positional embedding:")
print(hidden_state)

# Test output of hidden_state
print("\nExpected number of tiles per batch based on aspect ratios:")
for idx, (h, w) in enumerate(aspect_ratios):
    print(f"Batch {idx}: {h}x{w} = {h*w} tiles")
    num_tiles = h * w
    # Ensure only the valid tiles are updated
    print(f"Updated tiles in hidden state (non-zero entries): {torch.count_nonzero(hidden_state[idx, :num_tiles])}")
    assert torch.count_nonzero(hidden_state[idx, num_tiles:]) == 0, \
        f"Error: Non-updated tiles should remain zero for batch {idx}"
