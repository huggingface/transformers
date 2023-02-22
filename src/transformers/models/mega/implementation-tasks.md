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
* To modify
  * ~~`MegaEmbeddings` - add optional token type embeddings  ~~
  * ~~`MegaLayer` - start with something like this for the combined encoder/decoder layer~~
    * There's also a nice handling of cross-attention already
    * If `output_attentions`, do we want self-attention or cross attention? both? (both)
  * ~~`MegaEncoder`: pass through multiple mega layers~~
    * Might not be needed, but especially helps with using cached values between layers
    * Actually going to delete this, but here's what we should keep from it:
      * Final hidden states (each token from the last layer)
      * Tuple of updated decoder caches from each layer (if `use_cache`, otherwise `None`)
      * Layerwise hidden states (if `output_hidden_states`, otherwise None)
      * Self-attentions from each layer (if `output_attentions`, otherwise None)
      * Cross-attentions from each layer (if `output_attentions` and `config.add_cross_attention`, otherwise None)
    * **STATUS**: done
  * ~~`MegaModel`: pass through the embeddings and encoder~~
    * Should also set it up to work with `inputs_embeds` so that we can pass token embeddings directly (helpful for explainability tooling like gradient-based attribution)
    * Might need to name things like in the MegaLM class I used 
      * `embedding_layer` and `encoders`
    * Optional pooling layer
    * **STATUS**: finished
  * `MegaPretrainedModel` - initialization, remove checkpointing (?)
  * Downstream work: 
    * ~~Remove position_ids and head_mask everywhere~~
    * Documentation updates
* Decoder stuff:
  * Probably untangle the incremental state in OG Mega to look more like the `forward` method in the auto-generated `MegaEncoder` (which is meant to be the stack of N encoder or decoder layers)
* Initialization: once modules are ready to go into the `PretrainedModel` class, try to use its default initialization instead of one-off `reset_parameters` methods (which might be doing the same thing)

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
~~MultiHeadEMA~~: 
* return hidden state if `use_cache` (new input, in place of `incremental_state is not None`)
* use prior `state` value if provided (new input, possibly taken out of the `prev_key_values` tuple)
* **status:** done, pending testing within the wrapper classes

~~MovingAverageGatedAttention~~
* Accept previous keys and values from `prev_key_values` (new input, in place of incremental_state)
* Return a tuple in the style of the HF attention modules (layer output, attention weights, key, value, EMA state) with contents controlled by `output_attentions` and `use_cache` (new inputs)
* **status:** done

prev_key_values inputs for ^ will be expected as:

```
(self_key, self_value, self_ema_state, cross_key, cross_value)
```

~~GatedCrossAttention~~
* Accept previous cross-attention keys and values from `prev_key_values` (new input, in place of incremental_state)
* Return keys and values if `use_cache` (new input)
* What about `pidx`?
  * It's actually used to index the positional bias for the current step (to preserve relative positions when doing one-at-a-time decoding)
  * If it's provided, it asserts that the queries are length 1 (which would be the case in incremental decoding); can we proxy this another way?
  * Currently getting this from the cached self-attention keys (cached self sequence length + 1)
* Attention masks


~~Then delete the IncrementalState class entirely~~

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
    * this actually isn't the case, RoBERTa generates one automatically if decoding, combines with padding mask, and broadcasts to 3D
    * the 3d mask doesn't work for Mega, but I essentially did the same thing and passed them separately to the Mega blocks
* Incremental state (if doing incremental decoding)
* Encoder hidden states (if doing cross-attention)

## Final Checklist for Implementation
* ~~Initialization - either do this in the `MegaPretrainedModel` class or take it out of there and keep the `reset_parameters` methods~~
  * Maybe leave in for MultiHeadEMA, ScaleNorm, and RMSNorm?
    * All of those are `nn.Parameter`, whereas `MegaPretrainedModel._init_weights` is working on embeddings, linear layers, and layernorm
  * Ultimately i think this is probably fine to have duplicated -- it seems that the defaults match what we're doing where they collide (and go a little farther on embeddings and other linear layers), and the option in the config would take precedence if a user changed it
* ~~Load my MLM weights and test~~ (done!!!)
  * Works exactly like the old one - even with padding
* ~~Figure out why extra weights are being loaded/initialized~~
  * Looks like a mismatch in the names due to a utility function renaming keys `gamma` and `beta` in the state dictionary... (see `_load_state_dict_into_model` in [this link](https://huggingface.co/transformers/v4.7.0/_modules/transformers/modeling_utils.html))
  * Renamed the parameters in the Mega modules and original checkpoint from `beta -> b_param` and `gamma -> g_param`
* ~~Rewrite `convert_mega_original_pytorch_checkpoint_to_pytorch` to save the pretrained wikitext MLM~~
* Rewrite tests in `tests/models/mega/test_modeling_mega`
  * To keep:
    * shape testing (all classes)
    * value/slice testing: MLM (updated) and no head (updated)
  * To delete:
    * Slice testing for classification head (done)
    * LM head ignore keys (done)
  * New tests:
    * bidirectionality
    * with/without token types
    * sequence length > max positions
    * chunking
    * softmax and element attention
* Documentation: 
  * ~~Docstrings of individual classes~~
  * Main docstring
  * Config docstring
  * Anything else in the `mega.mdx` file
* ~~Tokenizer?~~
  * looks like autotokenizer points to robertatokenizer, so I think I'm going to leave it for now
  * Tests don't require tokenizer
* Test a Mega encoder-decoder model?
  * Wouldn't have any pretrained weights, but could help catch bugs
  * Alternatively, could just push ahead with what I have 
* Train a simple text-to-text model?
* Push model to hub
  * Include tokenizer
  * Test mask-filling widget
* Open PR