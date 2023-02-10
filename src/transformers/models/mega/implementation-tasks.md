# Adding Mega: Things to Do

## Configuration
* ~~Modify my config class (from the colab notebook) to work like the auto-generated one~~
* ~~Possibly add explicit `use_nffn` args as well instead of implying based on other args~~
* `is_decoder` and `add_cross_attention` are already part of the `PretrainedConfig` parent class
* Either remove the `initializer_range` arg, or use it to init weights
* Remove gradient checkpointing option?
  * Actually inherited from the `PretrainedConfig`, so probably just raise a warning if it's requested

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
    * If `output_attentions`, do we want self-attention or cross attention? both?
    * Should also set it up to work with `inputs_embeds` so that we can pass token embeddings directly (helpful for explainability tooling like gradient-based attribution)
  * `MegaPretrainedModel` - initialization, remove checkpointing (?)
* Decoder stuff:
  * Probably untangle the incremental state in OG Mega to look more like the `forward` method in the auto-generated `MegaEncoder` (which is meant to be the stack of N encoder or decoder layers)
  * However, it doesn't really have to be the same, and RoBERTa uses a gross tuple of (decoder key, decoder value, encoder key, encoder value)
  * It is interesting that the fairseq method is stateful, with the internal functions of `MultiheadEMA` modifying it directly
  * Need to look into how the original implementation instantiates and passes the incremental state


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
    * This is controlled in the RoBERTa code (and in GPT2) with `use_cache`

Encoder 
* Input IDs (or embeddings)
* Attention mask (for padding)
* (optional) token type IDs

Decoder
* Input IDs (self / target sequence; or embeddings)
* Attention mask (padding)
* Autoregressive attention mask (is this needed separately?)
* Incremental state (if doing incremental decoding)
* Encoder hidden states (if doing cross-attention)