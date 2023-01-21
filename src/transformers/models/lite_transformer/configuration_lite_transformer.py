from transformers import PretrainedConfig


class LiteTransformerConfig(PretrainedConfig):

    def __init__(self,
                 encoder_vocab_size,
                 encoder_pad_token_id,
                 encoder_branch_type,
                 encoder_kernel_size_list,
                 decoder_vocab_size,
                 decoder_pad_token_id,
                 decoder_branch_type,
                 decoder_kernel_size_list,
                 encoder_ffn_list,
                 decoder_ffn_list,
                 weight_softmax=True,
                 encoder_glu=True,
                 decoder_glu=True,
                 encoder_conv_linear=True,
                 decoder_conv_linear=True,
                 no_token_positional_embeddings=False,
                 share_decoder_input_output_embed=True,
                 ffn_init=None,
                 encoder_learned_position_embedding=False,
                 encoder_layers=6,
                 encoder_attention_heads=8,
                 encoder_normalize_before=False,
                 encoder_ffn_embed_dim=200,
                 decoder_learned_position_embedding=False,
                 decoder_hidden_size=200,
                 decoder_layers=6,
                 decoder_attention_heads=8,
                 decoder_normalize_before=False,
                 decoder_ffn_embed_dim=200,
                 weight_dropout=None,
                 tie_adaptive_weights=None,
                 adaptive_softmax_cutoff=None,
                 adaptive_softmax_dropout=0,
                 no_decoder_final_norm=None,
                 attention_dropout=0,
                 activation_dropout=0,
                 activation_fn='relu',
                 input_dropout=0.1,
                 encoder_hidden_size=200,
                 dropout=1,
                 max_source_position_embeddings=1024,
                 max_target_position_embeddings=1024,
                 share_all_embeddings=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder_vocab_size = encoder_vocab_size
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_pad_token_id = encoder_pad_token_id
        self.encoder_learned_position_embedding = encoder_learned_position_embedding
        self.encoder_branch_type = encoder_branch_type
        self.activation_dropout = activation_dropout
        self.share_decoder_input_output_embed = share_decoder_input_output_embed
        self.encoder_layers = encoder_layers
        self.encoder_ffn_list = encoder_ffn_list
        self.decoder_ffn_list = decoder_ffn_list
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_kernel_size_list = encoder_kernel_size_list
        self.encoder_normalize_before = encoder_normalize_before
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.decoder_vocab_size = decoder_vocab_size
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_pad_token_id = decoder_pad_token_id
        self.decoder_learned_position_embedding = decoder_learned_position_embedding
        self.decoder_branch_type = decoder_branch_type
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_kernel_size_list = decoder_kernel_size_list
        self.decoder_normalize_before = decoder_normalize_before
        self.decoder_ffn_embed_dim = decoder_ffn_embed_dim
        self.ffn_init = ffn_init
        self.weight_softmax = weight_softmax
        self.weight_dropout = weight_dropout
        self.encoder_conv_linear = encoder_conv_linear
        self.decoder_conv_linear = decoder_conv_linear
        self.encoder_glu = encoder_glu
        self.decoder_glu = decoder_glu
        self.dropout = dropout
        self.no_token_positional_embeddings = no_token_positional_embeddings
        self.tie_adaptive_weights = tie_adaptive_weights
        self.adaptive_softmax_cutoff = adaptive_softmax_cutoff
        self.adaptive_softmax_dropout = adaptive_softmax_dropout
        self.no_decoder_final_norm = no_decoder_final_norm
        self.attention_dropout = attention_dropout
        self.activation_fn = activation_fn
        self.input_dropout = input_dropout
        self.max_source_position_embeddings = max_source_position_embeddings
        self.max_target_position_embeddings = max_target_position_embeddings
        self.share_all_embeddings = share_all_embeddings

        if len(self.encoder_kernel_size_list) == 1:
            self.encoder_kernel_size_list = self.encoder_kernel_size_list * self.encoder_layers
        elif len(self.encoder_kernel_size_list) > self.encoder_layers:
            self.encoder_kernel_size_list = self.encoder_kernel_size_list[:self.encoder_layers]

        if len(self.decoder_kernel_size_list) == 1:
            self.decoder_kernel_size_list = self.decoder_kernel_size_list * self.decoder_layers
        elif len(self.decoder_kernel_size_list) > self.decoder_layers:
            self.decoder_kernel_size_list = self.decoder_kernel_size_list[:self.decoder_layers]

        if not self.weight_dropout:
            self.weight_dropout = self.attention_dropout

        self.encoder_ffn_list = self.encoder_ffn_list * self.encoder_layers if len(
            self.encoder_ffn_list) == 1 else self.encoder_ffn_list
        self.decoder_ffn_list = self.decoder_ffn_list * self.decoder_layers if len(
            self.decoder_ffn_list) == 1 else self.decoder_ffn_list
