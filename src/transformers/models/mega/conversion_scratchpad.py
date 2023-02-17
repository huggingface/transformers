# import new hugging face classes
from .configuration_mega import MegaConfig
from .modeling_mega import MegaForMaskedLM

# import 
import torch 
from torch import nn 

# import the EncoderLayer class used to pretrain
# !! NOTE !! this requires the version of fairseq that is built when you install the Mega source 
# steps: 
#   1. `git clone https://github.com/facebookresearch/mega.git`
#   2. `cd mega && pip install -e`
try:
    from fairseq.modules.mega_layer import MegaEncoderLayer
except:
    raise ImportError("You need to install the version of fairseq from the Mega repo!")

# import the model weights and config file
# we'll need to point to where you ran 
# `git clone https://huggingface.co/mnaylor/mega-wikitext-103`
ORIGINAL_WEIGHTS_DIR = '/users/Mitch/Documents/MiscProjects/mega-wikitext-103'
import os
import pickle as pkl

with open(os.path.join(ORIGINAL_WEIGHTS_DIR, 'model_args.pkl'), 'rb') as f:
    mega_original_args = pkl.load(f)

# the wrapper classes used to train the MLM  (see colab notebook below)
# https://colab.research.google.com/drive/1qfUO6o5HRdxBblWlw058HVyvaEPhPpH8?usp=sharing
# MegaLM outputs hidden states
class MegaLM(nn.Module):
  'The base class for our Mega encoder - given input IDs, embed text and return encoder output'
  def __init__(self, mega_args, depth, vocab_size):
    super().__init__()
    self.mega_args = mega_args
    self.embedding_layer = nn.Embedding(vocab_size, self.mega_args.encoder_embed_dim)
    self.encoders = nn.ModuleList(
      [MegaEncoderLayer(self.mega_args) for _ in range(depth)
    ])
    self.depth = depth
        
  def forward(self, input_ids, attention_mask, batch_first=True, ignore_mask_value=0):
    '''
    Code for a forward pass - expects input_ids and attention_mask to come
    from a Hugging Face tokenizer as PyTorch tensors, and returns a tensor
    of size (batch, n_classes) containing classification logits
    
    Other options:
      - batch_first: boolean indicating whether the batch dimension is first 
        in input_ids (default: True, which aligns with the HF tokenizer behavior)
      - ignore_mask_value: the value in attention_mask that identifies tokens 
        that should be ignored (default: 0, which aligns with HF tokenizer)
    '''

    # Mega expects embeddings to be (time, batch, embedding size), but 
    # Hugging Face returns tokens as (batch, time)
    if batch_first:
        input_ids = input_ids.T

    # to make things more confusing, Mega expects the attention mask to
    # be (batch, time), but with values of 0 (normal token) and 1 (ignore token)
    # which is the opposite of what HF returns
    if ignore_mask_value == 0:
        attention_mask = 1 - attention_mask

    # get token embeddings from IDs
    embeds = self.embedding_layer(input_ids)

    # pass through the Mega layers
    # input is (time, batch, encoder dim) and output is the same
    for encoder in self.encoders:
        embeds = encoder(embeds, attention_mask)
        
    # return according to the shape specified
    if batch_first:
        # (T, B, H) --> (B, T, H)
        return torch.transpose(embeds, 0, 1)
    else:
        return embeds

# renamed from MegaForMaskedLM to avoid confusion with new module
class OriginalMegaForMaskedLM(nn.Module):
  'A wrapper class for doing masked language modeling with Mega'
  def __init__(self, mega_args, depth, vocab_size):
    super().__init__()
    self.mega = MegaLM(mega_args, depth, vocab_size)
    self.mlm_head = nn.Linear(mega_args.encoder_embed_dim, vocab_size)
    self.dropout = nn.Dropout(p=0.1)

  def forward(self, input_ids, attention_mask, batch_first=True, ignore_mask_value=0):
    """
    Perform a forward pass through the Mega encoder and the masked LM head. Returns
    logits for each vocabulary entry.

    If `batch_first` (default to align with Hugging Face tokenizer behavior), 
    output will have the shape (Batch size, Sequence length, Vocab size);
    otherwise (S, B, V)
    """
    encoder_output = self.mega(input_ids, attention_mask, batch_first, ignore_mask_value)
    return self.mlm_head(self.dropout(encoder_output))

# load the original encoder
original_mlm = OriginalMegaForMaskedLM(**mega_original_args).eval()

# load its weights
print("Original Mega encoder:", original_mlm.mega.load_state_dict(torch.load(os.path.join(ORIGINAL_WEIGHTS_DIR, 'encoder_weights.pt'), map_location='cpu')))
print("Original Mega MLM layer:", original_mlm.mlm_head.load_state_dict(torch.load(os.path.join(ORIGINAL_WEIGHTS_DIR, 'mlm_head_weights.pt'), map_location='cpu')))

# create a new config from the old one
hf_config = MegaConfig(
    num_hidden_layers=mega_original_args['depth'],
    vocab_size=mega_original_args['vocab_size'],
    hidden_size=mega_original_args['mega_args'].encoder_embed_dim,
    shared_representation_size=mega_original_args['mega_args'].encoder_z_dim,
    intermediate_size=mega_original_args['mega_args'].encoder_hidden_dim,
    ema_projection_size=mega_original_args['mega_args'].encoder_n_dim,
    dropout_prob=mega_original_args['mega_args'].dropout,
    attention_probs_dropout_prob=mega_original_args['mega_args'].attention_dropout,
    hidden_dropout_prob=mega_original_args['mega_args'].hidden_dropout,
    activation=mega_original_args['mega_args'].activation_fn,
    attention_activation=mega_original_args['mega_args'].attention_activation_fn,
    bidirectional=mega_original_args['mega_args'].bidirectional,
    use_chunking = mega_original_args['mega_args'].encoder_chunk_size > 0,
    chunk_size=mega_original_args['mega_args'].encoder_chunk_size,
    truncation=mega_original_args['mega_args'].truncation_length,
    normalization_type=mega_original_args['mega_args'].normalization_type,
    normalize_before_mega=True,
    norm_affine=True,
    use_feature_dropout=mega_original_args['mega_args'].feature_dropout,
    relative_positional_bias=mega_original_args['mega_args'].rel_pos_bias,
    max_positions=mega_original_args['mega_args'].max_source_positions,
    nffn_hidden_size=mega_original_args['mega_args'].encoder_ffn_embed_dim,
    normalize_before_ffn=mega_original_args['mega_args'].normalize_before,
    nffn_activation_dropout_prob=0.0,
    add_token_type_embeddings=False
)

hf_mlm = MegaForMaskedLM(hf_config, add_dense_layer=False).eval()

hf_mlm.mega.embedding_layer.word_embeddings.weight = original_mlm.mega.embedding_layer.weight
print("HF Mega encoder:", hf_mlm.mega.encoders.load_state_dict(original_mlm.mega.encoders.state_dict()))
print("HF Mega MLM layer:", hf_mlm.mlm_head.load_state_dict(torch.load(os.path.join(ORIGINAL_WEIGHTS_DIR, 'mlm_head_weights.pt'), map_location='cpu')))

input_ids = torch.randint(0, hf_config.vocab_size, size=(4, 256))
input_mask = torch.ones_like(input_ids)
# mask a few tokens to make sure masking is applied appropriately
input_mask[:, -10:] = 0

original_output = original_mlm(input_ids, input_mask, batch_first=True, ignore_mask_value=0)
hf_output = hf_mlm(input_ids, input_mask)[0]

print(f"original output {original_output.shape}")
print(f"hf output {hf_output.shape}")
print(f"max diff: {(original_output - hf_output).max()}") # 0.0
assert torch.allclose(original_output, hf_output, atol=1e-3), f"Original:\n{original_output}\n\nHF\n{hf_output}"

hf_mlm.save_pretrained("/users/Mitch/Documents/MiscProjects/mega-base-wikitext")