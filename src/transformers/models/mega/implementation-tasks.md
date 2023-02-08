# Adding Mega: Things to Do

## Configuration
* Add proper config class 
* Add `is_decoder` to config (since encoder/decoder will use separate configs)
* Possibly explicit `use_nffn` and `use_cross_attention` args as well instead of implying based on other args

## Mega source code
* Possibly combine `MegaEncoderLayer` and `MegaDecoderLayer` into a single class
* Revisit how I'm doing embeddings (and how we should handle token type embeddings)

## Conversion to HF
* Possibly delete:
  * `MegaAttention` in favor of using the `MovingAverageGatedAttention` class
  * `MegaIntermediate`
  * `MegaOutput`
* To modify
  * `MegaLayer` - start with something like this for the combined encoder/decoder layer
  * `MegaPretrainedModel` - initialization, remove checkpointing (?)