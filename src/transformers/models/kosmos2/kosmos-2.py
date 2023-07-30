import re
import torch


# Get this from the original kosmos-2 demo
original_kosmos2_checkpoint_only_2_layers = "kosmos-2-state-dict-num-layers-2.bin"
dog_sample_file = "sample_dog.bin"
snowman_sample_file = "sample_snowman.bin"
snowman_sample_detail_file = "sample_snowman_detail.bin"


from transformers.models.kosmos2.configuration_kosmos2 import Kosmos2Config, Kosmos2VisionConfig
from transformers.models.kosmos2.modeling_kosmos2 import Kosmos2Model, Kosmos2ForConditionalGeneration


# conversion


def rename_vision_key(key):

    # text decoder
    key = re.sub(r"img_model.visual\.", "vision_model.model.", key)

    key = re.sub(r"\.class_embedding$", ".embeddings.class_embedding", key)
    key = re.sub(r"\.positional_embedding$", ".embeddings.position_embedding.weight", key)
    key = re.sub(r"\.conv1.weight$", ".embeddings.patch_embedding.weight", key)

    key = re.sub(r"\.ln_pre\.", ".pre_layrnorm.", key)


    key = re.sub(r"\.transformer.resblocks\.", ".encoder.layers.", key)

    key = re.sub(r"\.ts_attn\.", ".self_attn.", key)

    key = re.sub(r"\.ln_1\.", ".layer_norm1.", key)
    key = re.sub(r"\.ln_2\.", ".layer_norm2.", key)

    key = re.sub(r"\.c_fc\.", ".fc1.", key)
    key = re.sub(r"\.c_proj\.", ".fc2.", key)

    key = re.sub(r"\.ln_post\.", ".post_layernorm.", key)

    return key


def rename_key(key):
    # text decoder
    key = re.sub(r"gpt_model.decoder\.", "text_model.", key)
    # text decode: `embed_tokens`
    key = re.sub(r"\.embed_tokens\.", ".model.embed_tokens.", key)

    # text decode: `embed_positions` (no weight)
    # key: gpt_model.decoder.embed_positions._float_tensor
    # renamed_key: text_model.embed_positions._float_tensor

    key = re.sub(r"\.layers\.", ".model.layers.", key)

    key = re.sub(r"^text_model.layer_norm\.", "text_model.model.layer_norm.", key)

    key = re.sub(r"^text_model.output_projection\.", "text_model.lm_head.", key)

    key = re.sub(r"^img_connector\.", "image_to_text_connector.", key)

    # not used in forward!
    # self.self_attn_sope

    key = rename_vision_key(key)

    return key


# ==============================================================================================================
# Original model topology


"""
UniGPTmodel(
  (gpt_model): TransformerLanguageModel(
    (decoder): LMDecoder(
      (dropout_module): Dropout(p=0.1, inplace=True)
      (embed_tokens): Embedding(65037, 2048, padding_idx=1)
      (embed_positions): SinusoidalPositionalEmbedding()place=True)
      (output_projection): Linear(in_features=2048, out_features=65037, bias=False)
      (layers): ModuleList(features=2048, out_features=8192, bias=True)
        (0-23): 24 x DecoderLayer(s=8192, out_features=2048, bias=True)
          (dropout_module): Dropout(p=0.1, inplace=True)92]), eps=1e-05, elementwise_affine=True)
          (self_attn): MultiheadAttention(
            (k_proj): Linear(in_features=2048, out_features=2048, bias=True)ementwise_affine=True)
            (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
            (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
            (out_proj): Linear(in_features=2048, out_features=2048, bias=True)fine=True)
            (inner_attn_ln): FusedLayerNorm(torch.Size([2048]), eps=1e-05, elementwise_affine=True)
            (dropout_module): Dropout(p=0.1, inplace=True)
          )
          (self_attn_layer_norm): FusedLayerNorm(torch.Size([2048]), eps=1e-05, elementwise_affine=True)
          (ffn): FeedForwardNetwork(
            (activation_dropout_module): Dropout(p=0.0, inplace=True)
            (dropout_module): Dropout(p=0.1, inplace=True)
            (fc1): Linear(in_features=2048, out_features=8192, bias=True)
            (fc2): Linear(in_features=8192, out_features=2048, bias=True)
            (ffn_layernorm): FusedLayerNorm(torch.Size([8192]), eps=1e-05, elementwise_affine=True)
          )
          (final_layer_norm): FusedLayerNorm(torch.Size([2048]), eps=1e-05, elementwise_affine=True)
        )
      )
      (layer_norm): FusedLayerNorm(torch.Size([2048]), eps=1e-05, elementwise_affine=True)
      (self_attn_sope): SoPE()
    )
  )
  (img_model): ClipVisualOnly(
    (visual): VisualTransformer4Seq2Seq(
      (conv1): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
      (ln_pre): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (transformer): Transformer(
        (resblocks): ModuleList(
          (0-23): 24 x ResidualAttentionBlock(
            (attn): None
            (ts_attn): MultiheadAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (dropout_module): Dropout(p=0.0, inplace=True)
            )
            (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): Sequential(
              (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
              (gelu): QuickGELU()
              (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (ln_post): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
"""


# ==============================================================================================================
# Vision


# load `kosmos-2` img_model to HF clip vision
# (with `model.img_model` --> `clip.vision_model`)


def load_and_check_vision_model():

    # ================================================================================
    from transformers import AutoModel
    hf_clip_model = AutoModel.from_pretrained("openai/clip-vit-large-patch14")

    # ================================================================================
    # load (small) state dict
    state_dict = torch.load(original_kosmos2_checkpoint_only_2_layers)
    state_dict_keys = list(state_dict.keys())

    # ================================================================================
    # rename key

    renamed_keys = []
    for key in state_dict_keys:
        if key.startswith("img_model"):
            renamed_key = rename_vision_key(key)
            renamed_keys.append(renamed_key)

    # ================================================================================
    # keep only 2 layers in HF clip model

    hf_clip_vision_keys = []
    for key in hf_clip_model.state_dict().keys():
        if key.startswith("vision_model") and not any(f".layers.{x}." in key for x in range(2, 25)):
            # we need to change `vision_model` to `vision_model.model`, as HF CLIP's vision model
            # starts with the key `vision_model`, but our model start with `vision_model.model`.
            # Note this is necessary only for the CLIP vision model.
            key = key.replace("vision_model", "vision_model.model")
            hf_clip_vision_keys.append(key)

    # ================================================================================
    # check renamed keys

    print(set(hf_clip_vision_keys).difference(renamed_keys))
    assert(set(hf_clip_vision_keys).issubset(renamed_keys))


def load_and_check_model(model, ckpt_path):

    # ================================================================================
    # load (small) state dict
    state_dict = torch.load(ckpt_path)
    state_dict_keys = list(state_dict.keys())

    # ================================================================================
    # rename key

    renamed_state_dict_keys = [rename_key(k) for k in state_dict_keys]

    # ================================================================================
    # check renamed keys

    model_state_dict_keys = list(model.state_dict().keys())
    diff_keys = set(model_state_dict_keys).difference(renamed_state_dict_keys)
    print(diff_keys)

    # all HF model keys should be in the renamed keys from the original checkpoint
    assert set(model_state_dict_keys).issubset(renamed_state_dict_keys)

    # ================================================================================
    # Create new model state dict

    loaded_model_state_dict = {}
    for key in state_dict_keys:
        renamed_key = rename_key(key)
        loaded_model_state_dict[renamed_key] = state_dict[key]

    # ================================================================================
    # check weight loading

    model.load_state_dict(loaded_model_state_dict, strict=False)


def check_model_with_dummy_inputs(model):
    """
    The `original model` here is the original `kosmos-2` model with the first 2 layers in both its text and vision
    components.
    """

    # ================================================================================
    # check loaded text model outputs

    # --------------------------------------------------------------------
    # For original kosmos-2

    # dummy_input_ids = torch.arange(0, 0 + 71, device="cuda").unsqueeze(0).clone()
    # # dummy_input_ids = torch.arange(2, 2 + 71, device="cuda").unsqueeze(0).clone()

    # original_text_outputs = model.gpt_model.decoder(dummy_input_ids, features_only=False)

    # # (original_text_outputs[0] is `logits`)
    # print(original_text_outputs[0].shape)
    # print(original_text_outputs[0])
    """
    torch.Size([1, 71, 65037])

    tensor(
        [
            [
                [15.6060, -5.1565,  8.0064,  ..., -2.2804, -2.0610, -1.0114],
                [ 9.7421, -4.9860,  4.9630,  ..., -1.5206, -1.4377, -0.5475],
                [11.6581, -5.1816, 18.9927,  ..., -2.3973, -1.9231, -1.3065],
                ...,
                [10.6616, -4.7872,  5.0161,  ..., -2.1092, -1.6931, -1.6256],
                [10.9655, -4.8194,  5.6438,  ..., -1.5778, -0.9324, -0.3715],
                [ 9.8335, -4.9696,  4.6688,  ..., -2.2745, -1.7485, -1.7921],
            ]
        ],
    )
    """

    # --------------------------------------------------------------------
    # Ours

    # Including the padding token id `1` in `input_ids` to make sure everything work
    # (especially the positional embedding)
    dummy_input_ids = torch.arange(0, 0 + 71, device="cpu").unsqueeze(0).clone()
    # dummy_input_ids = torch.arange(2, 2 + 71, device="cpu").unsqueeze(0).clone()

    hf_outputs = model.text_model(dummy_input_ids)

    print(hf_outputs.logits.shape)
    print(hf_outputs.logits)

    # --------------------------------------------------------------------
    # sanity check

    assert list(hf_outputs.logits.shape) == [1, 71, 65037]

    expected_block_1 = torch.tensor(
        [
            [15.605998,  -5.156513,   8.00637  ],
            [ 9.742102,  -4.9859715,  4.9629903],
            [11.658097,  -5.1816287, 18.992657 ],
        ],
    )
    expected_block_2 = torch.tensor(
        [
            [-2.2804384,  -2.0610392,  -1.0114111],
            [-1.5205702,  -1.4377496,  -0.54751855],
            [-2.39729,    -1.9231234,  -1.3065333],
        ],
    )
    expected_block_3 = torch.tensor(
        [
            [10.661624,  -4.7872415,  5.016076 ],
            [10.965461,  -4.819447,   5.6437974],
            [ 9.833488,  -4.9696445,  4.668755 ],
        ],
    )
    expected_block_4 = torch.tensor(
        [
            [-2.1091897, -1.693118,  -1.6255622],
            [-1.5777528, -0.9324262, -0.3714557],
            [-2.2744524, -1.7484771, -1.7920786],
        ],
    )

    diff_1 = torch.max(torch.abs(hf_outputs.logits[0, :+3, :+3] - expected_block_1))
    diff_2 = torch.max(torch.abs(hf_outputs.logits[0, :+3, -3:] - expected_block_2))
    diff_3 = torch.max(torch.abs(hf_outputs.logits[0, -3:, :+3] - expected_block_3))
    diff_4 = torch.max(torch.abs(hf_outputs.logits[0, -3:, -3:] - expected_block_4))

    max_diff = torch.max(torch.tensor([diff_1, diff_2, diff_3, diff_4]))
    assert max_diff < 3e-5

    # ================================================================================
    # check loaded vision model outputs

    # --------------------------------------------------------------------
    # For original kosmos-2

    # img_size, num_channels, batch_size = 224, 3, 1

    # # vision 2 layers
    # model.img_model.visual.transformer.resblocks = model.img_model.visual.transformer.resblocks[:2]

    # dummy_pixel_values = torch.ones(size=(batch_size, num_channels, img_size, img_size), device="cuda")

    # # The original shape is `torch.Size([257, 1, 1024])`
    # original_vision_output = model.img_model.visual(dummy_pixel_values)

    # # Make the shape being `torch.Size([1, 257, 1024])`
    # original_vision_output = original_vision_output.transpose(1, 0)

    # print(original_vision_output.shape)
    # print(original_vision_output)

    # # pooled (wth the post layer norm)
    # print(original_vision_output[:, 0, :])
    """
    torch.Size([1, 257, 1024])

    tensor(
        [
            [
                [-0.0115,  0.0596, -0.1132,  ...,  0.1799, -0.3139,  0.2753],
                [ 0.0846,  0.6236, -0.4391,  ...,  1.1525, -0.1509,  0.4326],
                [ 0.1050,  0.5756, -0.4778,  ...,  0.6579, -0.2205,  0.3997],
                ...,
                [ 0.1787,  0.5295, -0.6168,  ..., -0.9372, -0.3680,  0.2211],
                [ 0.1823,  0.5258, -0.5524,  ..., -0.8929, -0.3346,  0.2515],
                [ 0.0861,  0.5844, -0.6572,  ..., -0.7107, -0.2946,  0.3093],
            ],
        ],
    )
    
    tensor([[-0.0115,  0.0596, -0.1132,  ...,  0.1799, -0.3139,  0.2753]])
    """

    # --------------------------------------------------------------------
    # Ours

    img_size = model.config.vision_config.image_size  # 224
    num_channels = model.config.vision_config.num_channels  # 3

    batch_size = 1

    dummy_pixel_values = torch.ones(size=(batch_size, num_channels, img_size, img_size), device="cpu")

    hf_vision_output = model.vision_model(dummy_pixel_values)
    # HF CLIP has `last_hidden_state` without through `post_layernorm`
    hf_vision_output = model.vision_model.model.post_layernorm(hf_vision_output.last_hidden_state)

    print(hf_vision_output.shape)
    print(hf_vision_output[:, 0, :])

    # --------------------------------------------------------------------
    # sanity check

    assert list(hf_vision_output.shape) == [1, 257, 1024]

    expected_block_1 = torch.tensor(
        [
            [-0.01148908,  0.05956455, -0.11317716],
            [0.08458844,  0.6235921,  -0.43905595],
            [ 0.10498603,  0.57555795, -0.47783917],
        ],
    )
    expected_block_2 = torch.tensor(
        [
            [0.17986113, -0.31393886, 0.2753428],
            [1.1525147,  -0.15090114, 0.43260202],
            [0.6578805,  -0.22051974, 0.39973533],
        ],
    )
    expected_block_3 = torch.tensor(
        [
            [ 0.17873019,  0.5295236,  -0.6167609],
            [ 0.18225193,  0.52584666, -0.55239016],
            [ 0.08613532,  0.58441633, -0.6572151],
        ],
    )
    expected_block_4 = torch.tensor(
        [
            [-0.93716234, -0.36800045, 0.22114123],
            [-0.8929372,  -0.3345873, 0.2515392 ],
            [-0.7106602,  -0.2945692, 0.30925298],
        ],
    )

    diff_1 = torch.max(torch.abs(hf_vision_output[0, :+3, :+3] - expected_block_1))
    diff_2 = torch.max(torch.abs(hf_vision_output[0, :+3, -3:] - expected_block_2))
    diff_3 = torch.max(torch.abs(hf_vision_output[0, -3:, :+3] - expected_block_3))
    diff_4 = torch.max(torch.abs(hf_vision_output[0, -3:, -3:] - expected_block_4))

    max_diff = torch.max(torch.tensor([diff_1, diff_2, diff_3, diff_4]))
    assert max_diff < 1e-5

    # ================================================================================
    # check the whole model

    # --------------------------------------------------------------------
    # For original kosmos-2

    # dummy_img_attn_mask = torch.cat((torch.ones(size=(1, 64)), torch.zeros(size=(1, 7))), dim=-1).to("cuda").bool()

    # # pass only text
    # original_model_output_only_text = model(src_tokens=dummy_input_ids, img_src_tokens=None)
    # print(original_model_output_only_text[0].shape)
    # print(original_model_output_only_text[0])
    """
    torch.Size([1, 71, 65037])

    tensor(
        [
            [
                [15.6060, -5.1565,  8.0064,  ..., -2.2804, -2.0610, -1.0114],
                [ 9.7421, -4.9860,  4.9630,  ..., -1.5206, -1.4377, -0.5475],
                [11.6581, -5.1816, 18.9927,  ..., -2.3973, -1.9231, -1.3065],
                ...,
                [10.6616, -4.7872,  5.0161,  ..., -2.1092, -1.6931, -1.6256],
                [10.9655, -4.8194,  5.6438,  ..., -1.5778, -0.9324, -0.3715],
                [ 9.8335, -4.9696,  4.6688,  ..., -2.2745, -1.7485, -1.7921],
            ]
        ],
    )
    """

    # # pass text and vision
    # original_model_output = model(src_tokens=dummy_input_ids, img_src_tokens=dummy_pixel_values, img_gpt_input_mask=dummy_img_attn_mask)
    # print(original_model_output[0].shape)
    # print(original_model_output[0])
    """
    torch.Size([1, 71, 65037])

    tensor(
        [
            [
                [ 4.8882, -4.3499,  5.4597,  ..., -2.2055, -1.6321, -1.0148],
                [ 4.4254, -4.2447,  5.7366,  ..., -1.8535, -1.4237, -0.7096],
                [ 4.4483, -4.2894,  5.5115,  ..., -2.3162, -1.6573, -1.1387],
                ...,
                [ 8.1921, -5.0712,  5.3592,  ..., -2.5887, -2.0496, -1.8316],
                [ 8.4758, -5.1724,  5.9626,  ..., -1.7432, -1.1267, -0.5763],
                [ 7.6652, -5.2538,  5.4017,  ..., -2.4623, -1.9893, -1.9341],
            ]
        ],
    )
    """

    # --------------------------------------------------------------------
    # Ours

    dummy_img_attn_mask = torch.cat((torch.ones(size=(1, 64)), torch.zeros(size=(1, 7))), dim=-1).to("cpu").bool()

    # pass only text
    model_output_only_text = model.text_model(
        # pixel_values=None,
        input_ids=dummy_input_ids,
        img_attn_mask=None,
    )
    logits_only_text = model_output_only_text.logits

    print(logits_only_text)

    # pass text and vision
    model_output = model(
        pixel_values=dummy_pixel_values,
        input_ids=dummy_input_ids,
        img_attn_mask=dummy_img_attn_mask
    )
    logits = model_output.logits
    img_features = model_output.image_features

    print(logits.shape)
    print(logits)
    print(img_features.shape)
    print(img_features)

    # --------------------------------------------------------------------
    # sanity check: text input only

    assert list(logits_only_text.shape) == [1, 71, 65037]

    expected_block_1 = torch.tensor(
        [
            [15.605998,   -5.156513,    8.00637],
            [ 9.742102,   -4.9859715,   4.9629903],
            [11.658097,   -5.1816287,  18.992657],
        ],
    )
    expected_block_2 = torch.tensor(
        [
            [-2.2804384,  -2.0610392, -1.0114111],
            [-1.5205702,  -1.4377496, -0.54751855],
            [-2.39729,    -1.9231234, -1.3065333 ],
        ],
    )
    expected_block_3 = torch.tensor(
        [
            [10.661624,   -4.7872415,   5.016076],
            [10.965461,   -4.819447,    5.6437974],
            [ 9.833488,   -4.9696445,   4.668755],
        ],
    )
    expected_block_4 = torch.tensor(
        [
            [-2.1091897,  -1.693118, -1.6255622 ],
            [-1.5777528,  -0.9324262, -0.3714557],
            [-2.2744524,  -1.7484771, -1.7920786 ],
        ],
    )

    diff_1 = torch.max(torch.abs(logits_only_text[0, :+3, :+3] - expected_block_1))
    diff_2 = torch.max(torch.abs(logits_only_text[0, :+3, -3:] - expected_block_2))
    diff_3 = torch.max(torch.abs(logits_only_text[0, -3:, :+3] - expected_block_3))
    diff_4 = torch.max(torch.abs(logits_only_text[0, -3:, -3:] - expected_block_4))

    max_diff = torch.max(torch.tensor([diff_1, diff_2, diff_3, diff_4]))
    assert max_diff < 3e-5

    # --------------------------------------------------------------------
    # sanity check: text + image inputs

    assert list(logits_only_text.shape) == [1, 71, 65037]

    expected_block_1 = torch.tensor(
        [
            [4.888153,  -4.3498607,  5.4596553],
            [ 4.4253945, -4.244659,   5.736647],
            [4.448264,  -4.289385,  5.5114775],
        ],
    )
    expected_block_2 = torch.tensor(
        [
            [-2.2055285, -1.6320639, -1.0147916],
            [-1.8535209, -1.4236742, -0.7096378],
            [-2.3161755, -1.6573074, -1.1387042],
        ],
    )
    expected_block_3 = torch.tensor(
        [
            [8.192094,  -5.0711703,  5.3592353],
            [8.475775,  -5.172369,   5.9625816],
            [7.6652,    -5.2538114,  5.4017296],
        ],
    )
    expected_block_4 = torch.tensor(
        [
            [-2.5886784, -2.0495563, -1.831612],
            [-1.7432401, -1.1266646, -0.5763364],
            [-2.4622574, -1.9892663, -1.9341019],
        ],
    )

    diff_1 = torch.max(torch.abs(logits[0, :+3, :+3] - expected_block_1))
    diff_2 = torch.max(torch.abs(logits[0, :+3, -3:] - expected_block_2))
    diff_3 = torch.max(torch.abs(logits[0, -3:, :+3] - expected_block_3))
    diff_4 = torch.max(torch.abs(logits[0, -3:, -3:] - expected_block_4))

    max_diff = torch.max(torch.tensor([diff_1, diff_2, diff_3, diff_4]))
    assert max_diff < 3e-5

# ==============================================================================================================


def check_model_with_dog_sample(model):

    # Attention! On the original kosmos-2 `demo`, keep only the first 2 layers in the vision model:
    #   self.model.models[0].img_model.visual.transformer.resblocks = self.model.models[0].img_model.visual.transformer.resblocks[:2]

    # --------------------------------------------------------------------
    # real input: (dog)

    sample = torch.load(dog_sample_file, map_location=torch.device('cpu'))

    pixel_values = sample["net_input"]["img_src_tokens"]
    # It's of shape [1, 1, 3, 224, 224]. Change it to `[1, 3, 224, 224]`
    pixel_values = pixel_values[0]

    input_ids = sample["net_input"]["src_tokens"]
    img_attn_mask = sample["net_input"]["img_gpt_input_mask"]
    # We need a `bool` value
    img_attn_mask = img_attn_mask.bool()

    # --------------------------------------------------------------------
    # `use_cache=False`

    model_output_no_cache = model(pixel_values=pixel_values, input_ids=input_ids, img_attn_mask=img_attn_mask, use_cache=False)

    logits_no_cache = model_output_no_cache.logits
    past_key_values_no_cache = model_output_no_cache.past_key_values
    image_features_no_cache = model_output_no_cache.image_features

    # --------------------------------------------------------------------
    # `use_cache=True` to get the initial `past_key_values`

    model_output = model(pixel_values=pixel_values, input_ids=input_ids, img_attn_mask=img_attn_mask, use_cache=True)

    logits = model_output.logits
    past_key_values = model_output.past_key_values
    image_features = model_output.image_features

    # --------------------------------------------------------------------
    # verify the results between with/without using `cache`

    assert past_key_values_no_cache is None
    assert past_key_values is not None

    assert torch.max(torch.abs(image_features - image_features_no_cache)) < 1e-12
    assert torch.max(torch.abs(logits - logits_no_cache)) < 1e-12

    # --------------------------------------------------------------------
    # check with the original kosmos-2 output: initial step (step 71 -> step 72)

    assert list(logits.shape) == [1, 71, 65037]

    expected_block_1 = torch.tensor(
        [
            [15.605998  , -5.156513  ,  8.00637],
            [ 8.577738  , -4.9635577 ,  7.6196694],
            [ 5.5543556 , -4.5773745 ,  4.523568],
        ],
    )
    expected_block_2 = torch.tensor(
        [
            [-2.2804384 , -2.0610392 , -1.0114111],
            [-2.2657313 , -1.9836413 , -1.3702303],
            [-1.2256985 , -1.2151622 , -1.9965916],
        ],
    )
    expected_block_3 = torch.tensor(
        [
            [ 7.4827657 , -5.6471753 ,  5.3313484],
            [ 6.3412886 , -4.821356  ,  5.9151964],
            [ 7.3028603 , -5.5100656 ,  6.581722],
        ],
    )
    expected_block_4 = torch.tensor(
        [
            [-2.835022  , -2.887678  , -1.3593428 ],
            [-1.830313  , -1.4463289 , -1.2882515 ],
            [-2.29154   , -1.9426216 , -0.93513656],
        ],
    )

    diff_1 = torch.max(torch.abs(logits[0, :+3, :+3] - expected_block_1))
    diff_2 = torch.max(torch.abs(logits[0, :+3, -3:] - expected_block_2))
    diff_3 = torch.max(torch.abs(logits[0, -3:, :+3] - expected_block_3))
    diff_4 = torch.max(torch.abs(logits[0, -3:, -3:] - expected_block_4))

    max_diff = torch.max(torch.tensor([diff_1, diff_2, diff_3, diff_4]))
    assert max_diff < 3e-5

    # --------------------------------------------------------------------
    # next step: without `past_key_values`

    new_input_ids = torch.cat((input_ids, torch.tensor([[9]], dtype=torch.long, device="cpu")), dim=1)
    new_img_attn_mask = torch.cat((img_attn_mask, torch.tensor([[False]], dtype=torch.bool, device="cpu")), dim=1)
    new_model_output = model(pixel_values=pixel_values, input_ids=new_input_ids, img_attn_mask=new_img_attn_mask)

    new_logits = new_model_output.logits
    new_past_key_values = new_model_output.past_key_values

    assert new_past_key_values is None

    print(new_logits[:, -1, :])

    # --------------------------------------------------------------------
    # next step: with `past_key_values`

    next_input_ids = torch.tensor([[9]], dtype=torch.long, device="cpu")
    # (no need to pass `pixel_values`) -> need to specify it or `image_features`
    next_pixel_values = None
    next_image_features = image_features
    next_img_attn_mask = None
    next_model_output = model(pixel_values=next_pixel_values, img_features=next_image_features, input_ids=next_input_ids, img_attn_mask=next_img_attn_mask, past_key_values=past_key_values, use_cache=True)

    next_logits = next_model_output.logits
    next_past_key_values = next_model_output.past_key_values

    assert next_past_key_values is not None

    print(next_logits[:, -1, :])

    # --------------------------------------------------------------------
    # verify the results between with/without using `past_key_values`

    max_diff = torch.max(torch.abs(new_logits[:, -1, :] - next_logits[:, -1, :]))
    print(max_diff)
    assert max_diff < torch.tensor(3e-5)

    # --------------------------------------------------------------------
    # check with the original kosmos-2 output: next step (step 72 -> step 73)

    assert list(next_logits.shape) == [1, 1, 65037]

    expected_block_1 = torch.tensor([[[ 7.6893177 , -5.576222  ,  6.5033607]]])
    expected_block_2 = torch.tensor([[[ -2.398699  , -2.1435356 , -0.98740137]]])

    diff_1 = torch.max(torch.abs(next_logits[0, 0, :+3] - expected_block_1))
    diff_2 = torch.max(torch.abs(next_logits[0, 0, -3:] - expected_block_2))

    max_diff = torch.max(torch.tensor([diff_1, diff_2]))
    assert max_diff < 3e-5

    # --------------------------------------------------------------------
    # generation

    expected_generation = [
        9,     9,     5,     5,     5,    10,    10,    10,     5,
        5,   106,   106,   106,     6,     6,     6,     8,     8,     8,
        6,     6,   106,   106,    10,    10,    42,    42,    42,    10,
        10,   106,   106,    19,    19,    19,     6,     6,    12,    12,
        12,    20,    20,    20,    12,    12,    10,    10,    12,    12,
        106,   106,    43,    43,    43,  2115,  2115,  2115,    43,    43,
        106,   106,    12,    12,
    ]

    # use `text_model` directly
    # with `past_key_values` being passed as the initialized
    # no need to pass `img_features` (`pixel_values`) and `img_attn_mask`
    generated_output = model.text_model.generate(
        # we need to provide the full `input_ids` not just the trailing one!
        input_ids=new_input_ids,
        use_cache=True,
        past_key_values=past_key_values,
        img_features=None,
        img_attn_mask=None,
        # we already generated the first token (step 71 -> 72)
        max_new_tokens=len(expected_generation) - 1,
    )

    # --------------------------------------------------------------------
    # check with the original kosmos-2 output generation output: (step 72 -> step X)

    assert generated_output[0, 71:].tolist() == expected_generation


def check_real_model_with_dog_sample(model):

    # --------------------------------------------------------------------
    # real input: (dog)

    sample = torch.load(dog_sample_file, map_location=torch.device('cpu'))

    pixel_values = sample["net_input"]["img_src_tokens"]
    # It's of shape [1, 1, 3, 224, 224]. Change it to `[1, 3, 224, 224]`
    pixel_values = pixel_values[0]

    input_ids = sample["net_input"]["src_tokens"]
    img_attn_mask = sample["net_input"]["img_gpt_input_mask"]
    # We need a `bool` value
    img_attn_mask = img_attn_mask.bool()

    # --------------------------------------------------------------------
    # `use_cache=False`

    model_output_no_cache = model(pixel_values=pixel_values, input_ids=input_ids, img_attn_mask=img_attn_mask, use_cache=False)

    logits_no_cache = model_output_no_cache.logits
    past_key_values_no_cache = model_output_no_cache.past_key_values
    image_features_no_cache = model_output_no_cache.image_features

    # --------------------------------------------------------------------
    # `use_cache=True` to get the initial `past_key_values`

    model_output = model(pixel_values=pixel_values, input_ids=input_ids, img_attn_mask=img_attn_mask, use_cache=True)

    logits = model_output.logits
    past_key_values = model_output.past_key_values
    image_features = model_output.image_features

    # --------------------------------------------------------------------
    # verify the results between with/without using `cache`

    assert past_key_values_no_cache is None
    assert past_key_values is not None

    assert torch.max(torch.abs(image_features - image_features_no_cache)) < 1e-12
    assert torch.max(torch.abs(logits - logits_no_cache)) < 1e-12

    # --------------------------------------------------------------------
    # check with the original kosmos-2 output: initial step (step 71 -> step 72)

    assert list(logits.shape) == [1, 71, 65037]

    expected_block_1 = torch.tensor(
        [
            [2.920926332473755, -5.4380574226379395, 2.8645224571228027],
            [0.004667307715862989, -5.0997819900512695, 4.338554382324219],
            [-0.5761765837669373, -4.547626972198486, 3.8142454624176025],
        ],
    )
    expected_block_2 = torch.tensor(
        [
            [-2.61974835395813, -2.6742029190063477, -1.6856958866119385],
            [-2.251966714859009, -2.242988348007202, -1.5341331958770752],
            [-2.3858885765075684, -1.5038200616836548, -1.013083577156067],
        ],
    )
    expected_block_3 = torch.tensor(
        [
            [-1.3929418325424194, -4.623406410217285, 3.7545101642608643],
            [0.522249698638916, -4.5460662841796875, 7.236062526702881],
            [-1.7789695262908936, -5.221266746520996, 3.770735740661621],
        ],
    )
    expected_block_4 = torch.tensor(
        [
            [-2.3952505588531494, -2.878037452697754, -1.3662471771240234],
            [-3.3000922203063965, -3.0199999809265137, -0.24584506452083588],
            [-2.8502795696258545, -3.096112012863159, -0.771698534488678],
        ],
    )

    diff_1 = torch.max(torch.abs(logits[0, :+3, :+3] - expected_block_1))
    diff_2 = torch.max(torch.abs(logits[0, :+3, -3:] - expected_block_2))
    diff_3 = torch.max(torch.abs(logits[0, -3:, :+3] - expected_block_3))
    diff_4 = torch.max(torch.abs(logits[0, -3:, -3:] - expected_block_4))

    max_diff = torch.max(torch.tensor([diff_1, diff_2, diff_3, diff_4]))
    assert max_diff < 3e-5

    expected_next_token = 64007
    predicted_next_token = torch.argmax(logits[0, -1, :]).detach().to("cpu").numpy().tolist()

    assert predicted_next_token == expected_next_token

    # --------------------------------------------------------------------
    # next step: without `past_key_values`

    next_token = expected_next_token

    new_input_ids = torch.cat((input_ids, torch.tensor([[next_token]], dtype=torch.long, device="cpu")), dim=1)
    new_img_attn_mask = torch.cat((img_attn_mask, torch.tensor([[False]], dtype=torch.bool, device="cpu")), dim=1)
    new_model_output = model(pixel_values=pixel_values, input_ids=new_input_ids, img_attn_mask=new_img_attn_mask)

    new_logits = new_model_output.logits
    new_past_key_values = new_model_output.past_key_values

    assert new_past_key_values is None

    print(new_logits[:, -1, :])

    # --------------------------------------------------------------------
    # next step: with `past_key_values`

    next_input_ids = torch.tensor([[next_token]], dtype=torch.long, device="cpu")
    # (no need to pass `pixel_values`) -> need to specify it or `image_features`
    next_pixel_values = None
    next_image_features = image_features
    next_img_attn_mask = None
    next_model_output = model(pixel_values=next_pixel_values, img_features=next_image_features, input_ids=next_input_ids, img_attn_mask=next_img_attn_mask, past_key_values=past_key_values, use_cache=True)

    next_logits = next_model_output.logits
    next_past_key_values = next_model_output.past_key_values

    assert next_past_key_values is not None

    print(next_logits[:, -1, :])

    # --------------------------------------------------------------------
    # verify the results between with/without using `past_key_values`

    max_diff = torch.max(torch.abs(new_logits[:, -1, :] - next_logits[:, -1, :]))
    print(max_diff)
    assert max_diff < torch.tensor(3e-5)

    # --------------------------------------------------------------------
    # check with the original kosmos-2 output: next step (step 72 -> step 73)

    assert list(next_logits.shape) == [1, 1, 65037]

    expected_block_1 = torch.tensor([[[-1.3323104, -5.1079516,  5.359114 ]]])
    expected_block_2 = torch.tensor([[[-2.8319776, -3.5213413, -1.8274367]]])

    diff_1 = torch.max(torch.abs(next_logits[0, 0, :+3] - expected_block_1))
    diff_2 = torch.max(torch.abs(next_logits[0, 0, -3:] - expected_block_2))

    max_diff = torch.max(torch.tensor([diff_1, diff_2]))
    assert max_diff < 3e-5

    expected_next_token = 94
    predicted_next_token = torch.argmax(next_logits[0, -1, :]).detach().to("cpu").numpy().tolist()

    assert predicted_next_token == expected_next_token

    # --------------------------------------------------------------------
    # repeat the above check with more extra steps (step 73 -> step 74)
    # check with the original kosmos-2 output: next step

    # steps: 73 -> 74 -> .. -> 83 (-> 84)
    steps = list(range(73, 83 + 1))

    next_tokens = [94, 17772, 64008, 64009, 64092, 65029, 64011, 64148, 65021, 64010, 1280, 12]

    expected_blocks = [
        ([-1.5333264, -5.0365257,  5.595204 ], [-2.1252668, -2.9195867, -1.3610152]),
        ([-1.14558  , -4.6416078,  8.611397 ], [-1.9524179, -2.3943331, -1.2364707]),
        ([ 2.1540604, -2.713409 ,  1.8866036], [ 1.631276 ,  0.8916559, -0.4697148]),
        ([-1.1833401, -3.1272492, -1.4443989], [2.75421  , 2.1421206, 1.2756062]),
        ([-1.429662 , -4.2857313,  1.123333 ], [13.215454, 13.476381, 14.000856]),
        ([-0.10423064, -4.0805306 ,  7.669438  ], [1.3264995 , 0.37444258, 2.872366  ]),
        ([-1.9969933, -4.391607 , -3.4535604], [ 0.15055317,  0.05899912, -0.0650674 ]),
        ([-2.1537013, -4.2108035,  2.163306 ], [8.790059, 8.622845, 9.70795 ]),
        ([-0.10536906, -2.7584782 ,  5.857536  ], [4.7097054, 3.5752287, 6.4874005]),
        ([-0.40761316, -4.65115   , 16.127958  ], [-3.0172224, -3.2040298, -2.283117 ]),
        ([ 0.4004581, -4.4891667, 14.7836075], [-0.9358875, -1.006671 , -0.1364981]),
    ]

    for (step, next_token, expected_next_token, (expected_block_1, expected_block_2)) in zip(steps, next_tokens[:-1], next_tokens[1:], expected_blocks):

        print(f"step: {step}")

        new_input_ids = torch.cat((new_input_ids, torch.tensor([[next_token]], dtype=torch.long, device="cpu")), dim=1)
        new_img_attn_mask = torch.cat((new_img_attn_mask, torch.tensor([[False]], dtype=torch.bool, device="cpu")), dim=1)
        new_model_output = model(pixel_values=pixel_values, input_ids=new_input_ids, img_attn_mask=new_img_attn_mask)

        new_logits = new_model_output.logits
        new_past_key_values = new_model_output.past_key_values

        assert new_past_key_values is None

        print(new_logits[:, -1, :])

        # --------------------------------------------------------------------
        # next step: with `past_key_values`

        next_input_ids = torch.tensor([[next_token]], dtype=torch.long, device="cpu")
        # (no need to pass `pixel_values`) -> need to specify it or `image_features`
        next_pixel_values = None
        next_image_features = image_features
        next_img_attn_mask = None
        next_model_output = model(pixel_values=next_pixel_values, img_features=next_image_features, input_ids=next_input_ids, img_attn_mask=next_img_attn_mask, past_key_values=next_past_key_values, use_cache=True)

        next_logits = next_model_output.logits
        next_past_key_values = next_model_output.past_key_values

        assert next_past_key_values is not None

        print(next_logits[:, -1, :])

        # --------------------------------------------------------------------
        # verify the results between with/without using `past_key_values`

        max_diff = torch.max(torch.abs(new_logits[:, -1, :] - next_logits[:, -1, :]))
        # step 75 has a slightly bigger diff
        allowed_max_diff = 3e-5 if step != 75 else 5e-5

        assert max_diff < torch.tensor(allowed_max_diff)

        # --------------------------------------------------------------------
        # check with the original kosmos-2 output: next step

        assert list(next_logits.shape) == [1, 1, 65037]

        expected_block_1 = torch.tensor([[expected_block_1]])
        expected_block_2 = torch.tensor([[expected_block_2]])

        diff_1 = torch.max(torch.abs(next_logits[0, 0, :+3] - expected_block_1))
        diff_2 = torch.max(torch.abs(next_logits[0, 0, -3:] - expected_block_2))

        max_diff = torch.max(torch.tensor([diff_1, diff_2]))
        allowed_max_diff = 3e-5

        assert max_diff < allowed_max_diff

        predicted_next_token = torch.argmax(next_logits[0, -1, :]).detach().to("cpu").numpy().tolist()

        assert predicted_next_token == expected_next_token

    # --------------------------------------------------------------------
    # generation

    new_input_ids = torch.cat((new_input_ids, torch.tensor([[predicted_next_token]], dtype=torch.long, device="cpu")), dim=1)
    new_img_attn_mask = torch.cat((new_img_attn_mask, torch.tensor([[False]], dtype=torch.bool, device="cpu")), dim=1)

    expected_generation = [
         64007,    94, 17772, 64008, 64009, 64092, 65029, 64011, 64148,
         65021, 64010,  1280,    12, 64007,     5,  4464, 64008, 64009, 64013,
         65036, 64010, 2
    ]

    # use `text_model` directly
    # with `past_key_values` being passed as the initialized
    # no need to pass `img_features` (`pixel_values`) and `img_attn_mask`
    generated_output = model.text_model.generate(
        input_ids=new_input_ids,
        use_cache=True,
        past_key_values=next_past_key_values,
        img_features=None,
        img_attn_mask=None,
        # we already generated 13 tokens: from `64007` (step 71 -> 72) to `12` (step 83 -> 84)
        max_new_tokens=len(expected_generation) - 13,
    )

    # --------------------------------------------------------------------
    # check with the original kosmos-2 output generation output: (step 84 -> step X)

    assert generated_output[0, 71:].tolist() == expected_generation

    # --------------------------------------------------------------------
    # Use `eos_token_id`

    # or we can specify `eos_token_id` to stop earlier.
    generated_output = model.text_model.generate(
        input_ids=new_input_ids,
        use_cache=True,
        past_key_values=next_past_key_values,
        img_features=None,
        img_attn_mask=None,
        # we already generated 13 tokens: from `64007` (step 71 -> 72) to `12` (step 83 -> 84)
        # use `eos_token_id=2` to stop earlier
        # TODO: specify this in the config file
        # we still need to specify this: so we get long enough generations
        max_new_tokens=len(expected_generation),
        eos_token_id=2,
    )
    assert generated_output[0, 71:].tolist() == expected_generation

    # --------------------------------------------------------------------
    # generation without `use_cache` (from step 84)

    # use `text_model` directly
    # with `use_cache=False` and `past_key_values=None`
    # need to pass `img_features` and `img_attn_mask` (for the `correctness`)
    generated_output = model.text_model.generate(
        input_ids=new_input_ids,
        use_cache=False,
        past_key_values=None,
        img_features=image_features,
        img_attn_mask=new_img_attn_mask,
        # we already generated 13 tokens: from `64007` (step 71 -> 72) to `12` (step 83 -> 84)
        max_new_tokens=len(expected_generation) - 13,
    )
    assert generated_output[0, 71:].tolist() == expected_generation

    # --------------------------------------------------------------------
    # generation without `use_cache` (from the start)

    # use `text_model` directly
    # with`use_cache=False` (from the start --> `past_key_values=None`)
    # need to pass `img_features` and `img_attn_mask` (for the `correctness`)
    generated_output = model.text_model.generate(
        input_ids=input_ids,
        use_cache=False,
        past_key_values=None,
        img_features=image_features,
        img_attn_mask=img_attn_mask,
        max_new_tokens=len(expected_generation),
        output_scores=True,
        return_dict_in_generate=True,
    )

    assert generated_output.sequences[0, 71:].tolist() == expected_generation

    for score, expected_block in zip(generated_output.scores[2:12], expected_blocks):
        assert torch.max(torch.abs(score[0, :+3] - torch.tensor(expected_block[0]))) < 5e-5
        assert torch.max(torch.abs(score[0, -3:] - torch.tensor(expected_block[1]))) < 5e-5

    # --------------------------------------------------------------------
    # generation with `use_cache` (from the start)

    # use `text_model` directly
    # with `use_cache=True` (from the start --> `past_key_values=None`)
    # need to pass `img_features` and `img_attn_mask` (for the `correctness`)
    generated_output = model.text_model.generate(
        input_ids=input_ids,
        use_cache=True,
        past_key_values=None,
        img_features=image_features,
        img_attn_mask=img_attn_mask,
        max_new_tokens=len(expected_generation),
        output_scores=True,
        return_dict_in_generate=True,
    )

    assert generated_output.sequences[0, 71:].tolist() == expected_generation

    for score, expected_block in zip(generated_output.scores[2:12], expected_blocks):
        assert torch.max(torch.abs(score[0, :+3] - torch.tensor(expected_block[0]))) < 3e-5
        assert torch.max(torch.abs(score[0, -3:] - torch.tensor(expected_block[1]))) < 3e-5

    # --------------------------------------------------------------------
    # generation without `use_cache` (from the start)

    # use `model`
    # with`use_cache=False` (from the start --> `past_key_values=None`)
    generated_output = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        use_cache=False,
        past_key_values=None,
        # we can specify `None` here.
        img_features=image_features,
        img_attn_mask=img_attn_mask,
        max_new_tokens=len(expected_generation),
        output_scores=True,
        return_dict_in_generate=True,
    )

    assert generated_output.sequences[0, 71:].tolist() == expected_generation

    for score, expected_block in zip(generated_output.scores[2:12], expected_blocks):
        assert torch.max(torch.abs(score[0, :+3] - torch.tensor(expected_block[0]))) < 5e-5
        assert torch.max(torch.abs(score[0, -3:] - torch.tensor(expected_block[1]))) < 5e-5

    # --------------------------------------------------------------------
    # generation with `use_cache` (from the start)

    # use `model`
    # with `use_cache=True` (from the start --> `past_key_values=None`)
    generated_output = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        use_cache=True,
        past_key_values=None,
        img_features=None,
        img_attn_mask=img_attn_mask,
        max_new_tokens=len(expected_generation),
        output_scores=True,
        return_dict_in_generate=True,
    )

    assert generated_output.sequences[0, 71:].tolist() == expected_generation

    for score, expected_block in zip(generated_output.scores[2:12], expected_blocks):
        assert torch.max(torch.abs(score[0, :+3] - torch.tensor(expected_block[0]))) < 3e-5
        assert torch.max(torch.abs(score[0, -3:] - torch.tensor(expected_block[1]))) < 3e-5

    # --------------------------------------------------------------------
    # generation with `use_cache` (from the start)

    # use `model`
    # with `use_cache=True` (from the start --> `past_key_values=None`)
    generated_output = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        use_cache=True,
        past_key_values=None,
        img_features=image_features,
        img_attn_mask=img_attn_mask,
        max_new_tokens=len(expected_generation),
        output_scores=True,
        return_dict_in_generate=True,
    )

    assert generated_output.sequences[0, 71:].tolist() == expected_generation

    for score, expected_block in zip(generated_output.scores[2:12], expected_blocks):
        assert torch.max(torch.abs(score[0, :+3] - torch.tensor(expected_block[0]))) < 3e-5
        assert torch.max(torch.abs(score[0, -3:] - torch.tensor(expected_block[1]))) < 3e-5

    # --------------------------------------------------------------------
    # generation with `use_cache` (from step 84)

    # use `model`
    # with `use_cache=True`
    generated_output = model.generate(
        pixel_values=pixel_values,
        input_ids=new_input_ids,
        use_cache=True,
        past_key_values=next_past_key_values,
        img_features=None,
        img_attn_mask=new_img_attn_mask,
        # we already generated 13 tokens: from `64007` (step 71 -> 72) to `12` (step 83 -> 84)
        max_new_tokens=len(expected_generation) - 13,
    )
    assert generated_output[0, 71:].tolist() == expected_generation

    # --------------------------------------------------------------------
    # generation with `use_cache` (from step 84)

    # use `model`
    # with `use_cache=True`
    generated_output = model.generate(
        pixel_values=pixel_values,
        input_ids=new_input_ids,
        use_cache=True,
        past_key_values=next_past_key_values,
        img_features=image_features,
        img_attn_mask=new_img_attn_mask,
        # we already generated 13 tokens: from `64007` (step 71 -> 72) to `12` (step 83 -> 84)
        max_new_tokens=len(expected_generation) - 13,
    )
    assert generated_output[0, 71:].tolist() == expected_generation


def check_real_model_with_snowman_sample(model):

    # --------------------------------------------------------------------
    # real input: (snowman)

    sample = torch.load(snowman_sample_file, map_location=torch.device('cpu'))

    pixel_values = sample["net_input"]["img_src_tokens"]
    # It's of shape [1, 1, 3, 224, 224]. Change it to `[1, 3, 224, 224]`
    pixel_values = pixel_values[0]

    input_ids = sample["net_input"]["src_tokens"]
    img_attn_mask = sample["net_input"]["img_gpt_input_mask"]
    # We need a `bool` value
    img_attn_mask = img_attn_mask.bool()

    # --------------------------------------------------------------------
    # generation with `use_cache`

    expected_generation = [64007, 10, 43867, 64008, 64009, 64057, 64876, 64010, 5950, 597, 32, 64007, 10, 646, 64008, 64009, 64018, 64924, 64010, 4, 2]

    # use `model`
    # with `use_cache=True`
    generated_output = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        use_cache=True,
        past_key_values=None,
        img_features=None,
        img_attn_mask=img_attn_mask,
        max_new_tokens=len(expected_generation),
    )
    assert generated_output[0, 71:].tolist() == expected_generation


def check_real_model_with_snowman_detail_sample(model):

    # --------------------------------------------------------------------
    # real input: (snowman detail)

    sample = torch.load(snowman_sample_detail_file, map_location=torch.device('cpu'))

    pixel_values = sample["net_input"]["img_src_tokens"]
    # It's of shape [1, 1, 3, 224, 224]. Change it to `[1, 3, 224, 224]`
    pixel_values = pixel_values[0]

    input_ids = sample["net_input"]["src_tokens"]
    img_attn_mask = sample["net_input"]["img_gpt_input_mask"]
    # We need a `bool` value
    img_attn_mask = img_attn_mask.bool()

    # --------------------------------------------------------------------
    # generation with `use_cache`

    expected_generation = [
        24,  1648,  1338,    10, 43867,  1280,
        32, 64007,    10, 30879, 64008, 64009, 64018, 65020, 64010,    12,
        5,  1842,     4,    71,    17,  1679, 64007,    10,  3958, 64008,
        64009, 64061, 64263, 64010,     6, 64007, 15719, 64008, 64009, 64253,
        64617, 64010,     6,     8, 64007,  9626, 64008, 64009, 64413, 64545,
        64010,     6,    23, 64007,    10,  4363, 64008, 64009, 64623, 64885,
        64010,  2255,     8, 64007,    10,  3486, 64008, 64009, 64809, 65036,
        64010,  1560,  2255,     4,    24, 43867,  1684,     7,    27,  3774,
        5, 10356,     9,     5,   646,     6,     8,    22,  1684,     7,
        30,    10,  2007,     8, 16239,  4337,     4,     2
    ]

    # use `model`
    # with `use_cache=True`
    generated_output = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        use_cache=True,
        past_key_values=None,
        img_features=None,
        img_attn_mask=img_attn_mask,
        max_new_tokens=len(expected_generation),
    )
    assert generated_output[0, 75:].tolist() == expected_generation
    # generated_output = model.generate(pixel_values=pixel_values, input_ids=input_ids, use_cache=True, past_key_values=None, img_features=None, img_attn_mask=img_attn_mask, max_new_tokens=94)


def check_head_base_model_loading(config):

    model = Kosmos2ForConditionalGeneration(config=config)
    ckpt = "Kosmos2ForConditionalGeneration"
    model.save_pretrained(ckpt)

    loaded_config = Kosmos2Config.from_pretrained(ckpt)
    loaded_config.architectures = ["Kosmos2Model"]
    loaded_config.save_pretrained(ckpt)
    loaded_config = Kosmos2Config.from_pretrained(ckpt)
    print(loaded_config.architectures)

    _ = Kosmos2Model.from_pretrained(ckpt)


    base_model = Kosmos2Model(config=config)
    ckpt = "Kosmos2Model"
    base_model.save_pretrained(ckpt)

    loaded_config = Kosmos2Config.from_pretrained(ckpt)
    loaded_config.architectures = ["Kosmos2ForConditionalGeneration"]
    loaded_config.save_pretrained(ckpt)
    loaded_config = Kosmos2Config.from_pretrained(ckpt)
    print(loaded_config.architectures)

    _ = Kosmos2ForConditionalGeneration.from_pretrained(ckpt)


def create_model(num_layers=2):

    text_config = {
        "use_cache": False,
        "scale_embedding": True,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "activation_function": "gelu",
        "activation_dropout": 0.0,
        "add_cross_attention": False,
        "attention_heads": 32,
        "ffn_dim": 8192,
        "embed_dim": 2048,
        "layers": num_layers,
        "layer_norm_eps": 1e-5,
        "gradient_checkpointing": False,
        # to match the demo
        "no_repeat_ngram_size": 3,
        # "use_cache": True,

    }
    vision_config = Kosmos2VisionConfig()
    #  2 layers
    vision_config.num_hidden_layers = num_layers
    vision_config = vision_config.to_dict()

    latent_query_num = 64

    config = Kosmos2Config(text_config=text_config, vision_config=vision_config, latent_query_num=latent_query_num)
    model = Kosmos2ForConditionalGeneration(config=config)
    model.eval()

    print(model)

    return model


if __name__ == "__main__":

    # ================================================================================
    # check tokenizer and processor

    from src.transformers.models.kosmos2.processing_kosmos2 import Kosmos2Processor
    from src.transformers.models.kosmos2.tokenization_kosmos2 import Kosmos2Tokenizer
    from src.transformers.models.kosmos2.tokenization_kosmos2_fast import Kosmos2TokenizerFast
    from transformers import CLIPImageProcessor

    slow_tokenizer = Kosmos2Tokenizer(vocab_file="sentencepiece.bpe.model")
    fast_tokenizer = Kosmos2TokenizerFast(__slow_tokenizer=slow_tokenizer)
    image_processor = CLIPImageProcessor()
    slow_processor = Kosmos2Processor(tokenizer=slow_tokenizer, image_processor=image_processor)
    fast_processor = Kosmos2Processor(tokenizer=fast_tokenizer, image_processor=image_processor)
    print(slow_processor)

    r1 = slow_tokenizer.tokenize("I love <phrase>this dog</phrase>")
    # ['▁I', '▁love', '<phrase>', '▁this', '▁dog', '</phrase>']
    print(r1)
    r1 = slow_tokenizer("I love <phrase>this dog</phrase>")
    # {'input_ids': [0, 13, 275, 64007, 38, 1133, 64008, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
    print(r1)

    # for fast tokenizer, we will get extra token if a tag token has a space before it.
    # we also get an extra token before "this" token.
    r2 = fast_tokenizer.tokenize("I love <phrase>this dog</phrase>")
    # ['▁I', '▁love', '▁', '<phrase>', 'this', '▁dog', '</phrase>']
    print(r2)
    r2 = fast_tokenizer("I love <phrase>this dog</phrase>")
    # {'input_ids': [0, 13, 275, 106, 64007, 5966, 1133, 64008, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
    print(r2)

    # Avoid this issue by `removing that space` + `having a space before a normal token that is after a tag token`
    r3 = fast_tokenizer.tokenize("I love<phrase> this dog</phrase>")
    # ['▁I', '▁love', '<phrase>', '▁this', '▁dog', '</phrase>']
    print(r3)
    r3 = fast_tokenizer("I love<phrase> this dog</phrase>")
    # {'input_ids': [0, 13, 275, 64007, 38, 1133, 64008, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}
    print(r3)

    texts = ["<grounding>I love  <phrase> this dog</phrase> and <phrase>those 2 cats </phrase>"]
    # Fake value
    image = True
    bboxes = [[(1, 2), [(1, 64), (7, 100)]]]

    r4 = slow_processor.preprocess_text(texts=texts, images=image, bboxes=bboxes)
    print(r4)

    r5 = fast_processor.preprocess_text(texts=texts, images=image, bboxes=bboxes)
    print(r5)

    from PIL import Image
    image = Image.open("dog_cats.jpg")

    r6 = slow_processor(text=texts, images=image, bboxes=bboxes)
    print(r6)

    r7 = fast_processor(text=texts, images=image, bboxes=bboxes)
    print(r7)

    generated_ids = [
        64012,   712,  1648,     9, 64007,    94, 17772, 64008, 64009, 64092, 65029, 64011, 64148, 65021,
        64010,  1280,    12, 64007,     5,  4464, 64008, 64009, 64013, 65036, 64010,     2
    ]
    r8 = fast_processor.decode(generated_ids)
    print(r8)

    # slow_tokenizer.push_to_hub("ydshieh/kosmos-2-patch14-224", use_auth_token="XXX")
    # fast_tokenizer.push_to_hub("ydshieh/kosmos-2-patch14-224", use_auth_token="XXX")
    # image_processor.push_to_hub("ydshieh/kosmos-2-patch14-224", use_auth_token="XXX")
    # fast_processor.push_to_hub("ydshieh/kosmos-2-patch14-224", use_auth_token="XXX")

    # fast_processor = Kosmos2Processor.from_pretrained("ydshieh/kosmos-2-patch14-224")
    # r9 = fast_processor.decode(generated_ids)
    # print(r9)

    text = "<grounding>Describe this image in detail:"
    image = Image.open("snowman.jpg")
    bboxes = None

    # There is a big problem if the tag token is at the beginning of the sentence
    inputs = fast_processor(text=text, images=image, bboxes=bboxes)
    print(inputs)

    exit(0)

    # ================================================================================
    # config & model creation

    dummy_model = create_model(num_layers=2)

    # ================================================================================
    # check the head model's checkpoint could be loaded into the base model and vice-versa

    check_head_base_model_loading(dummy_model.config)

    # ================================================================================
    # check model keys and loading

    load_and_check_vision_model()
    load_and_check_model(dummy_model, ckpt_path=original_kosmos2_checkpoint_only_2_layers)

    # ================================================================================
    # check loaded text model outputs

    # Tip:
    # We need to pass `attention mask` if we want to call decoder layers directly!
    # (use `_prepare_decoder_attention_mask`)

    # Tip
    # Including the padding token id `1`  in `input_ids` to make sure everything work
    # (especially the positional embedding)

    check_model_with_dummy_inputs(dummy_model)
    check_model_with_dog_sample(dummy_model)

    # ================================================================================
    # real config & model creation

    real_model = create_model(num_layers=24)

    # need to create this checkpoint
    load_and_check_model(real_model, ckpt_path="kosmos2_state_dict.bin")
    real_model.save_pretrained("HF_Kosmos2")

    # check we can load
    real_model = Kosmos2ForConditionalGeneration.from_pretrained("HF_Kosmos2")

    # # If we want to push to the Hub
    # repo_id = "ydshieh/kosmos-2-patch14-224"
    # real_model.save_pretrained("HF_Kosmos2", push_to_hub=True, repo_id=repo_id, use_auth_token="XXX")
    #
    # # check we can load from the Hub
    # real_model = Kosmos2ForConditionalGeneration.from_pretrained(repo_id)

    repo_id = "ydshieh/kosmos-2-patch14-224"

    # check we can load from the Hub
    real_model = Kosmos2ForConditionalGeneration.from_pretrained(repo_id)

    # ================================================================================

    #check_real_model_with_dog_sample(real_model)

    # ================================================================================

    #check_real_model_with_snowman_sample(real_model)

    # ================================================================================

    check_real_model_with_snowman_detail_sample(real_model)

    # ================================================================================