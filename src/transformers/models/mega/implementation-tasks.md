# Adding Mega: Things to Do

## Configuration
* Modify my config class (from the colab notebook) to work like the auto-generated one
* Possibly add explicit `use_nffn` args as well instead of implying based on other args
* `is_decoder` and `add_cross_attention` are already part of the `PretrainedConfig` parent class
* Either remove the `initializer_range` arg, or use it to init

## Mega source code
* Possibly combine `MegaEncoderLayer` and `MegaDecoderLayer` into a single class
* Revisit how I'm doing embeddings (and how we should handle token type embeddings)

## Conversion to HF
* Possibly delete these auto-generated modules:
  * `MegaAttention` in favor of using the `MovingAverageGatedAttention` class
    * Could possibly wrap up cross-attention and self attention in a `MegaAttention` class
  * `MegaIntermediate` (unnecessary)
  * `MegaOutput`
* To modify
  * `MegaEmbeddings` - add optional token type embeddings  
  * `MegaLayer` - start with something like this for the combined encoder/decoder layer
    * There's also a nice handling of cross-attention already
    * If `output_attentions`, do we want self-attention or cross attention? both?
  * `MegaPretrainedModel` - initialization, remove checkpointing (?)