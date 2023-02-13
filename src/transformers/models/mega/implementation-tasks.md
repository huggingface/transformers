# Adding Mega: Things to Do

## Configuration
* ~~Modify my config class (from the colab notebook) to work like the auto-generated one~~
* ~~Possibly add explicit `use_nffn` args as well instead of implying based on other args~~
* `is_decoder` and `add_cross_attention` are already part of the `PretrainedConfig` parent class
* Either remove the `initializer_range` arg, or use it to init weights
  * Actual initialization via config is happening in `MegaPretrainedModel`, so need to pick that or the original inits
* Remove gradient checkpointing option?
  * ~~Actually inherited from the `PretrainedConfig`, so probably just raise a warning if it's requested~~
  * Actually did this by setting `supports_gradient_checkpointing=False` in `MegaPretrainedModel`

## Mega source code
* Possibly combine `MegaEncoderLayer` and `MegaDecoderLayer` into a single class
* ~~Revisit how I'm doing embeddings (and how we should handle token type embeddings)~~

## Conversion to HF
* Possibly delete these auto-generated modules:
  * `MegaAttention` in favor of using the `MovingAverageGatedAttention` class
    * Could possibly wrap up cross-attention and self attention in a `MegaAttention` class, but probably not necessary as it's handled in the Mega block anyway
  * ~~`MegaIntermediate` (unnecessary)~~
  * `MegaOutput`
  * `create_position_ids
* To modify
  * `MegaEmbeddings` - add optional token type embeddings  
  * `MegaLayer` - start with something like this for the combined encoder/decoder layer
    * There's also a nice handling of cross-attention already
    * If `output_attentions`, do we want self-attention or cross attention? both? (both)
    * Should also set it up to work with `inputs_embeds` so that we can pass token embeddings directly (helpful for explainability tooling like gradient-based attribution)
  * `MegaPretrainedModel` - initialization, remove checkpointing (?)
* Decoder stuff:
  * Probably untangle the incremental state in OG Mega to look more like the `forward` method in the auto-generated `MegaEncoder` (which is meant to be the stack of N encoder or decoder layers)
  * However, it doesn't really have to be the same, and RoBERTa uses a gross tuple of (decoder key, decoder value, encoder key, encoder value)
    * No, we actually should set it up to return the tuple if `use_cache` is specified
  * It is interesting that the fairseq method is stateful, with the internal functions of `MultiheadEMA` modifying it directly
  * Need to look into how the original implementation instantiates and passes the incremental state
    * Expected shape in terms of batch / tgt seq len / hidden size?
    * Initial values?
    * Differences from HF's `past_key_values`

## Incremental State --> past_key_values
### Hugging Face's `past_key_values`
tuple of tuples whose entries are the attention keys and values for each decoder layer. these are the last outputs of the `layer_module` if `layer_module.is_decoder` stacked according to layer

```
past_key_values = (
    (layer_1_self_k, layer_1_self_v, layer_1_cross_k, layer_1_cross_v),
    (layer_2_self_k, layer_2_self_v, layer_2_cross_k, layer_2_cross_v),
    ...
    (layer_n_self_k, layer_n_self_v, layer_n_cross_k, layer_n_cross_v)
)
```

According to the docs for `BaseModelOutputWithPastAndCrossAttentions`, the shape of the tensors in the tuples is `(batch_size, num_heads, sequence_length, embed_size_per_head)`. Even though `num_heads` will always be 1 in our case, I'm guessing we'll need to follow the same format. This is the class used for RoBERTa's encoder output, though the specifics of `past_key_values` looks slightly different at the lower levels (i.e. only the single tuple-of-tensors is passed for a single layer).

Unlike the fairseq incremental state, the attention mask is combined with the standard self-attention mask. 

### Fairseq's Incremental State
Dictionary-like, with keys named according to the use of the values (all tensors). These tensors are stored with shape `(batch_size, sequence_length, embed_size_per_head)` - this is good!! We can probably just return instead of modifying the stateful representation. The way HF attention modules handle this is by returning a tuple of the following:
* context output (i.e. the weighted representation)
* attention probabilities (if requested by `output_attentions`; normalized by softmax)
* key layer output (if requested by `is_decoder`)
* value layer output (if requested by `is_decoder`)

It appears that incremental decoding begins with an empty IncrementalState (i.e. an empty dictionary), so that the first step populates the state. We can control this by using `if use_cache` where they use `if incremental_state is not None`.

Classes with incremental states:
* MultiHeadEMA (state; just a step-wise hidden state of the EMA portion)
* GatedCrossAttention (key, value, mask, number of steps (unused))
* MovingAverageGatedAttention (key, value, mask)
* MegaDecoderLayer (passed to the above layers)

This is mostly pretty standard: cross-attention takes cross-attn keys and values, self-attention takes self-attn keys and values. However, the only incremental state value `MultiHeadEMA` expects is something called `prev_state` -- this is just the incremental progress of the EMA local attention, and we'll need to incorporate that in our `prev_key_values`. It probably makes sense to have that stored as another item in the `prev_key_values` tuples (we'll only have 1 because it's self-only).

### Planning the change
MultiHeadEMA: 
* return hidden state if `use_cache` (new input, in place of `incremental_state is not None`)
* use prior `state` value if provided (new input, possibly taken out of the `prev_key_values` tuple)

MovingAverageGatedAttention
* Accept previous keys and values from `prev_key_values` (new input, in place of incremental_state)
* Return a tuple in the style of the HF attention modules (layer output, attention weights, key, value) with contents controlled by `output_attentions` and `use_cache` (new inputs)

GatedCrossAttention
* Accept previous cross-attention keys and values from `prev_key_values` (new input, in place of incremental_state)
* Return keys and values if `use_cache` (new input)

MegaDecoder
* Accept `prev_key_values` and `use_cache` (new input) and pass relevant information along to component modules

## Arguments needed for forward pass

Mega Block
* Input hidden states / embeds for the self sequence
* Attention mask for the self sequence
* If cross-attention:
  * Encoder hidden states 
  * Encoder attention mask
* If decoding:
  * Incremental state (previous attention keys, values, and padding mask)
    * note that this could be used for both cross-attention and unidirectional self-attention
    * But only necessary if doing incremental decoding (not necessarily the case, such as causal LM i.e. GPT)
    * This is controlled in the RoBERTa code (and in GPT2) with `use_cache` - which specifically determines whether to _return_ the attention information

Encoder 
* Input IDs (or embeddings)
* Attention mask (for padding)
* (optional) token type IDs

Decoder
* Input IDs (self / target sequence; or embeddings)
* Attention mask
  * Hugging Face combines padding masks with causal LM masks
* Incremental state (if doing incremental decoding)
* Encoder hidden states (if doing cross-attention)