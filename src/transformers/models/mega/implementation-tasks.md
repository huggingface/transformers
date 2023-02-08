# Adding Mega: Things to Do

## Configuration
* Modify my config class (from the colab notebook) to work like the auto-generated one
* Add `is_decoder` to config (since encoder/decoder will use separate configs)
* Possibly add explicit `use_nffn` and `use_cross_attention` args as well instead of implying based on other args

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
  * `MegaLayer` - start with something like this for the combined encoder/decoder layer
    * There's also a nice handling of cross-attention already
  * `MegaPretrainedModel` - initialization, remove checkpointing (?)