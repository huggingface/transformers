from transformers import LlavaConfig, LlavaForConditionalGeneration, PixtralProcessor, MistralConfig, PixtralConfig, PreTrainedTokenizerFast, PixtralImageProcessor

import torch
from safetensors.torch import load_file as safe_load_file
import regex as re

from PIL import Image
import requests
from transformers import AutoProcessor



from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

# Load Mistral tokenizer

model_name = "mistralai/Pixtral-12B-2409"

tokenizer = MistralTokenizer.from_model(model_name)

vocab = tokenizer.instruct_tokenizer.tokenizer._tekken_token2id_nospecial
all_special = [token.value if hasattr(token,"value") else token for token in tokenizer.instruct_tokenizer.tokenizer._all_special_tokens]
specials_tokens = {token : all_special.index(token)  for token in all_special}
specials_tokens.update(vocab)
vocab = specials_tokens
from transformers.convert_slow_tokenizer import *
class MistralConverter:
    """
    A general tiktoken converter.
    """

    def __init__(
        self,
        vocab=None,
        pattern=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        add_prefix_space=False,
        additional_special_tokens=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args)
        self.vocab = vocab
        self.pattern = pattern
        self.add_prefix_space = add_prefix_space
        self.additional_special_tokens = additional_special_tokens

    def extract_vocab_merges_from_model(self, vocab: str):
        try:
            from tiktoken.load import load_tiktoken_bpe
        except Exception:
            raise ValueError(
                "`tiktoken` is required to read a `tiktoken` file. Install it with " "`pip install tiktoken`."
            )

        bpe_ranks = vocab 
        byte_encoder = bytes_to_unicode()

        def token_bytes_to_string(b):
            return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

        merges = []
        vocab = {}
        for idx, (token, rank) in enumerate(bpe_ranks.items()):
            if token not in all_special:
                vocab[token_bytes_to_string(token)] = idx
                if len(token) == 1:
                    continue
                local = []
                for index in range(1, len(token)):
                    piece_l, piece_r = token[:index], token[index:]
                    if piece_l in bpe_ranks and piece_r in bpe_ranks and (piece_l + piece_r) in bpe_ranks:
                        local.append((piece_l, piece_r, rank))
                local = sorted(local, key=lambda x: (bpe_ranks[x[0]], bpe_ranks[x[1]]), reverse=False)
                merges.extend(local)
            else:
                vocab[token] = idx
        merges = sorted(merges, key=lambda val: val[2], reverse=False)
        merges = [(token_bytes_to_string(val[0]), token_bytes_to_string(val[1])) for val in merges]
        return vocab, merges

    def tokenizer(self):
        vocab_scores, merges = self.extract_vocab_merges_from_model(self.vocab)
        tokenizer = Tokenizer(BPE(vocab_scores, merges, fuse_unk=False))
        if hasattr(tokenizer.model, "ignore_merges"):
            tokenizer.model.ignore_merges = True
        return tokenizer

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(self.pattern), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=self.add_prefix_space, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.add_special_tokens(self.additional_special_tokens)

        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        return tokenizer

tokenizer = PreTrainedTokenizerFast(tokenizer_object = MistralConverter(vocab=vocab, additional_special_tokens=all_special).converted())


text_config = MistralConfig(
    attention_dropout=0.0,
    bos_token_id=1,
    eos_token_id=2,
    head_dim=128,
    hidden_act="silu",
    hidden_size=5120,
    initializer_range=0.02,
    intermediate_size=14336,
    max_position_embeddings=1024000,
    model_type="mistral",
    num_attention_heads=32,
    num_hidden_layers=40,
    num_key_value_heads=8,
    rms_norm_eps=1e-05,
    rope_theta=1000000000.0,
    sliding_window=None,
    tie_word_embeddings=False,
    vocab_size=131072
)

vision_config = PixtralConfig()
config = LlavaConfig(vision_config, text_config)
config.architectures = ["LlavaForConditionalGeneration"]
config.text_config.head_dim = 128
config.save_pretrained("../pixtral")

tokenizer.model_input_names = ['input_ids', 'attention_mask']
original_state_dict = safe_load_file("../pixtral/consolidated.safetensors")


OLD_KEY_TO_NEW_KEY_MAPPING = {
    # Layer Normalization Weights
    r"vision_encoder.transformer.layers.(\d+).input_layernorm.weight":  r"vision_tower.transformer.layers.\1.attention_norm.weight",
    r"vision_encoder.transformer.layers.(\d+).ffn_norm.weight":         r"vision_tower.transformer.layers.\1.ffn_norm.weight",
    
    # Self Attention Projections
    r"vision_encoder.transformer.layers.(\d+).attention.wq.weight":     r"vision_tower.transformer.layers.\1.attention.q_proj.weight",
    r"vision_encoder.transformer.layers.(\d+).attention.wk.weight":     r"vision_tower.transformer.layers.\1.attention.k_proj.weight",
    r"vision_encoder.transformer.layers.(\d+).attention.wv.weight":     r"vision_tower.transformer.layers.\1.attention.v_proj.weight",
    r"vision_encoder.transformer.layers.(\d+).attention.wo.weight":     r"vision_tower.transformer.layers.\1.attention.o_proj.weight",
    
    # MLP Projections
    r"vision_encoder.transformer.layers.(\d+).feed_forward.w1.weight":  r"vision_tower.transformer.layers.\1.feed_forward.gate_proj.weight",
    r"vision_encoder.transformer.layers.(\d+).feed_forward.w2.weight":  r"vision_tower.transformer.layers.\1.feed_forward.down_proj.weight",
    r"vision_encoder.transformer.layers.(\d+).feed_forward.w3.weight":  r"vision_tower.transformer.layers.\1.feed_forward.up_proj.weight",
    
    # Additional mappings
    r"vision_encoder":                                  r"vision_tower",
    r"vision_language_adapter.w_in":                    r"multi_modal_projector.linear_1",
    r"vision_language_adapter.w_out":                   r"multi_modal_projector.linear_2",
    r"layers.(\d+).attention.wq.weight":                r"language_model.model.layers.\1.self_attn.q_proj.weight",
    r"layers.(\d+).attention.wk.weight":                r"language_model.model.layers.\1.self_attn.k_proj.weight",
    r"layers.(\d+).attention.wv.weight":                r"language_model.model.layers.\1.self_attn.v_proj.weight",
    r"layers.(\d+).attention.wo.weight":                r"language_model.model.layers.\1.self_attn.o_proj.weight",
    r"layers.(\d+).feed_forward.w1.weight":             r"language_model.model.layers.\1.mlp.gate_proj.weight",
    r"layers.(\d+).feed_forward.w2.weight":             r"language_model.model.layers.\1.mlp.down_proj.weight",
    r"layers.(\d+).feed_forward.w3.weight":             r"language_model.model.layers.\1.mlp.up_proj.weight",
    r"layers.(\d+).ffn_norm.weight":                    r"language_model.model.layers.\1.post_attention_layernorm.weight",
    r"layers.(\d+).attention_norm.weight":              r"language_model.model.layers.\1.input_layernorm.weight",
    r"tok_embeddings.weight":                           r"language_model.model.embed_tokens.weight",
    r"output.weight":                                   r"language_model.lm_head.weight",
    r"norm.weight":                                     r"language_model.model.norm.weight"

}



def permute_for_rope(value, n_heads, config):
        dim1 = value.shape[0]
        dim2 = config.hidden_size
        return value.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2) 

def convert_dictionnary(original_state_dict):
    new_dict={}

    all_keys = "\n"+ "\n".join(original_state_dict.keys())
    old_keys = all_keys
    for old, new in OLD_KEY_TO_NEW_KEY_MAPPING.items():
        all_keys = re.sub(r"\n"+ old,r"\n"+new,all_keys)

    OLD_TO_NEW = dict(zip(old_keys.split("\n"), all_keys.split("\n")))

    for key, value in original_state_dict.items():

        new_key = OLD_TO_NEW[key]
        if "vision_encoder" in key:
            _config = vision_config
            num_attention_heads = _config.num_attention_heads
        else:
            _config = text_config
            if "q_proj" in new_key:
                num_attention_heads = _config.num_attention_heads
            if "k_proj" in new_key:
                num_attention_heads = _config.num_key_value_heads
            # convert the text model (basically mistral model)


        if "q_proj" in new_key or "k_proj" in new_key:
            value = permute_for_rope(value,num_attention_heads, _config)

        new_dict[new_key] = value

config.text_config.head_dim = 128
# with torch.device("meta"):
#     model = LlavaForConditionalGeneration(config)
# model.load_state_dict(new_dict, strict=True, assign=True)

# model.save_pretrained("../pixtral")
config.vision_feature_layer = -1
config.image_token_index = 10
config.vision_feature_select_strategy = "full"
config.image_seq_length = 1
model = LlavaForConditionalGeneration.from_pretrained("../pixtral", config=config).to("cuda")
image_processor = PixtralImageProcessor()
processor = PixtralProcessor(tokenizer=tokenizer, image_processor=image_processor, image_token = "[IMG]")
processor.tokenizer = tokenizer
prompt = "[INST]\nWhat's the content of the image?"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")



messages = [
    {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": prompt}]
    },
]

tok = MistralTokenizer.from_model(model_name)


from mistral_common.protocol.instruct.request import ChatCompletionRequest
tokenized = tok.encode_chat_completion(
    ChatCompletionRequest(
        messages=messages,
        model=model_name,
    )
)

inputs["input_ids"] = torch.tensor([tokenized.tokens], dtype=torch.long, device="cuda")
inputs["pixel_values"] = torch.tensor(tokenized.images,  device="cuda")
del inputs["attention_mask"]
generate_ids = model.generate(**inputs, max_new_tokens=100)
print(processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
"""
What's the content of the image?The image depicts a vibrant street scene in what appears to be a Chinatown district, characterized by its traditional architectural elements and cultural signage. A prominent feature is the red and white stop sign in the foreground, which has been adorned with a banner that reads "OPTUS." Behind the stop sign, there's an ornate gate with intricate designs and Chinese characters, marking the entrance to the district. The gate is flanked by buildings with colorful facades and signs in both English and Chinese
"""