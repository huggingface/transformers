# from ...modeling_utils import PreTrainedModel

from transformers.models.lite_transformer.modeling_lite_transformer import LiteTransformerModel
from transformers.models.lite_transformer.configuration_lite_transformer import LiteTransformerConfig

config = LiteTransformerConfig(encoder_vocab_size=44512, encoder_pad_token_id=1,
                               encoder_branch_type=['attn:1:100:4',
                                                    'dynamic:default:100:4'],
                               encoder_kernel_size_list=[3, 7, 15, 31, 31, 31],
                               decoder_vocab_size=44512,
                               decoder_pad_token_id=1,
                               decoder_branch_type=['attn:1:100:4', 'dynamic:default:100:4'],
                               decoder_kernel_size_list=[3, 7, 15, 31, 31, 31],
                               encoder_ffn_list=[True, True, True, True, True, True],
                               decoder_ffn_list=[True, True, True, True, True, True]
                               )

model = LiteTransformerModel(config)

print(model)

