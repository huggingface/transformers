# coding=utf-8
# Copyright 2023 ylacombe The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Converting Meta SeamlessM4T checkpoints from seamless_communication to HF."""


import argparse
import os
from pathlib import Path

import torch
from accelerate.utils.modeling import find_tied_parameters
from huggingface_hub import HfApi
from seamless_communication.models.inference.translator import Translator

from transformers.models.seamless_m4t.configuration_seamless_m4t import SeamlessM4TConfig
from transformers.models.seamless_m4t.modeling_seamless_m4t import SeamlessM4TModel
from transformers.models.seamless_m4t.tokenization_seamless_m4t import SeamlessM4TTokenizer
from transformers.models.seamless_m4t.feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor

from transformers.trainer_utils import set_seed
from transformers.utils import logging


api = HfApi()


def assert_param_count(model_1, model_2):
    count_1 = sum(p[1].numel() for p in model_1.named_parameters() if "final_proj" not in p[0])
    count_2 = sum(p[1].numel() for p in model_2.named_parameters() if "final_proj" not in p[0])
    assert count_1 == count_2, f"{model_1.__class__}: {count_1} != {model_2.__class__}: {count_2}"


def param_count(model):
    return sum(p[1].numel() for p in model.named_parameters() if "final_proj" not in p[0])


def _grab_best_device(use_gpu=True):
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

vocoder_convert_list = [
    ("ups", "upsampler"),
    ("lang","lang_embeds_layer"),
    ("spkr","spkr_embeds_layer"),
    ("dict.","unit_embeds_layer."),
]

# order is important
wav2vec_convert_list = [
    ("speech_encoder_frontend.model_dim_proj", "feature_projection.projection"),
    ("speech_encoder_frontend.post_extract_layer_norm", "feature_projection.layer_norm"),
    ("speech_encoder_frontend.pos_encoder.conv", "encoder.pos_conv_embed.conv"),
    ("speech_encoder.inner.layers", "encoder.layers"),
    ("speech_encoder.inner_layer_norm", "encoder.layer_norm"),
    ("speech_encoder.adaptor_layers", "adapter.layers"),
    ("inner_proj", "intermediate_dense"),
    ("self_attn.output_proj", "self_attn.linear_out"),
    # ("self_attn.output_dense", "self_attn.linear_out"),
    ("output_proj", "output_dense"),
    ("self_attn.k_proj", "self_attn.linear_k"),
    ("self_attn.v_proj", "self_attn.linear_v"),
    ("self_attn.q_proj", "self_attn.linear_q"),
    ("self_attn.sdpa.u_bias", "self_attn.pos_bias_u"),
    ("self_attn.sdpa.v_bias", "self_attn.pos_bias_v"),
    ("self_attn.sdpa.r_proj", "self_attn.linear_pos"),
    ("conv.pointwise_conv1", "conv_module.pointwise_conv1"),
    ("conv.pointwise_conv2", "conv_module.pointwise_conv2"),
    ("conv.depthwise_conv", "conv_module.depthwise_conv"),
    ("conv.batch_norm", "conv_module.batch_norm"),
    ("conv_layer_norm", "conv_module.layer_norm"),
    ("speech_encoder.proj", "proj"),
    ("speech_encoder.layer_norm", "inner_layer_norm"),
    # "layer_norm", "encoder.layers.*.final_layer_norm",
    # "inner.layer_norm", "encoder.layer_norm",
]

t2u_convert_list = [
    ("t2u_model.final_proj", "lm_head"),
    ("t2u_model.", "model."),
    ("encoder_decoder_attn_layer_norm", "cross_attention_layer_norm"),
    ("encoder_decoder_attn", "cross_attention"),
    ("linear_k", "k_proj"),
    ("linear_v", "v_proj"),
    ("linear_q", "q_proj"),
    ("ffn.inner_proj", "ffn.fc1"),
    ("ffn.output_proj", "ffn.fc2"),
    ("output_proj", "out_proj"),
    ("decoder_frontend.embed", "decoder.embed_tokens"),
]

text_convert_list = [
    ("text_encoder.", ""),
    ("text_decoder.", ""),
    ("text_encoder_frontend.embed", "embed_tokens"),
    ("text_decoder_frontend.embed", "embed_tokens"),
    ("encoder_decoder_attn_layer_norm", "cross_attention_layer_norm"),
    ("encoder_decoder_attn", "cross_attention"),
    ("linear_k", "k_proj"),
    ("linear_v", "v_proj"),
    ("linear_q", "q_proj"),
    ("ffn.inner_proj", "ffn.fc1"),
    ("ffn.output_proj", "ffn.fc2"),
    ("output_proj", "out_proj"),
    ("final_proj", "lm_head"),
]

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
CACHE_DIR = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache_dir), "huggingface", "hub")


SAVE_DIR = "/home/ubuntu/weights"


def _load_original_model(device, name="seamlessM4T_medium"):
    unity_hub = Translator(name, "vocoder_36langs", device, torch.float32)

    return unity_hub


def _load_langs(model_type="medium"):
    if model_type == "medium":
        # fmt: off
        langs = ["ace","ace_Latn","acm","acq","aeb","afr","ajp","aka","amh","apc","arb","ars","ary","arz","asm","ast","awa","ayr","azb","azj","bak","bam","ban","bel","bem","ben","bho","bjn","bjn_Latn","bod","bos","bug","bul","cat","ceb","ces","cjk","ckb","crh","cym","dan","deu","dik","dyu","dzo","ell","eng","epo","est","eus","ewe","fao","pes","fij","fin","fon","fra","fur","fuv","gla","gle","glg","grn","guj","hat","hau","heb","hin","hne","hrv","hun","hye","ibo","ilo","ind","isl","ita","jav","jpn","kab","kac","kam","kan","kas","kas_Deva","kat","knc","knc_Latn","kaz","kbp","kea","khm","kik","kin","kir","kmb","kon","kor","kmr","lao","lvs","lij","lim","lin","lit","lmo","ltg","ltz","lua","lug","luo","lus","mag","mai","mal","mar","min","mkd","plt","mlt","mni","khk","mos","mri","zsm","mya","nld","nno","nob","npi","nso","nus","nya","oci","gaz","ory","pag","pan","pap","pol","por","prs","pbt","quy","ron","run","rus","sag","san","sat","scn","shn","sin","slk","slv","smo","sna","snd","som","sot","spa","als","srd","srp","ssw","sun","swe","swh","szl","tam","tat","tel","tgk","tgl","tha","tir","taq","taq_Tfng","tpi","tsn","tso","tuk","tum","tur","twi","tzm","uig","ukr","umb","urd","uzn","vec","vie","war","wol","xho","ydd","yor","yue","cmn","cmn_Hant","zul",]
        # fmt: on
        return langs
    else:
        # fmt: off
        langs = ["afr","amh","arb","ary","arz","asm","azj","bel","ben","bos","bul","cat","ceb","ces","ckb","cmn","cmn_Hant","cym","dan","deu","ell","eng","est","eus","fin","fra","fuv","gaz","gle","glg","guj","heb","hin","hrv","hun","hye","ibo","ind","isl","ita","jav","jpn","kan","kat","kaz","khk","khm","kir","kor","lao","lit","lug","luo","lvs","mai","mal","mar","mkd","mlt","mni","mya","nld","nno","nob","npi","nya","ory","pan","pbt","pes","pol","por","ron","rus","sat","slk","slv","sna","snd","som","spa","srp","swe","swh","tam","tel","tgk","tgl","tha","tur","ukr","urd","uzn","vie","yor","yue","zlm","zul",]
        # fmt: on
        return langs


def _load_hf_config(model_type="medium"):
    if model_type == "medium":
        # (model_dim=1024, w2v2_encoder_config=Wav2Vec2EncoderConfig(feature_dim=160, use_fbank=True, first_pass_dropout_p=0.0, layer_norm_features=False, feature_extractor_layer_descs=[], feature_extractor_bias=False, feature_extractor_layer_norm_convs=False, feature_grad_scale=0,pos_encoder_type='relative', pos_encoder_depth=0, pos_conv_kernel_size=0, num_pos_conv_groups=0, use_conformer=True, ffn_inner_dim=4096, dropout_p=0.0, attn_dropout_p=0.0, layer_drop_p=0.0, norm_order=<TransformerNormOrder.POST: 0>, depthwise_conv_kernel_size=31), nllb_config=NllbConfig(model_dim=1024, max_seq_len=1024,, pad_idx=0,dropout_p=0.1), t2u_config=UnitYT2UConfig(model_dim=1024, unit_max_seq_len=2048, unit_pad_idx=1, num_encoder_layers=4, num_decoder_layers=4, num_encoder_attn_heads=16, num_decoder_attn_heads=16, ffn_inner_dim=8192, dropout_p=0.1), use_text_encoder=True, use_conformer_adaptor=False, num_adaptor_layers=1, adaptor_kernel_size=8, adaptor_stride=8, adaptor_layer_norm=True, adaptor_dropout_p=0.1)
        kwargs = {
            "vocab_size": 256206,
            "unit_vocab_size": 10082,
            "hidden_size": 1024,
            "max_position_embeddings": 4096,
            "encoder_layers": 12,
            "decoder_layers": 12,
            "encoder_ffn_dim": 4096,
            "decoder_ffn_dim": 4096,
            "t2u_encoder_layers": 4,
            "t2u_decoder_layers": 4,
            "speech_encoder_layers": 12,
        }
        return SeamlessM4TConfig(**kwargs)
    else:
        return SeamlessM4TConfig()


def _convert_model(
    original_model,
    hf_model,
    convert_list,
    device,
    unwanted_prefix="model.",
    filter_state_dict="speech",
    exclude_state_dict=None,
):
    state_dict = original_model.state_dict()

    # filter func
    if isinstance(filter_state_dict, str):

        def filter_func(x):
            return filter_state_dict in x[0]

    else:

        def filter_func(item):
            if exclude_state_dict is not None and exclude_state_dict in item[0]:
                return False
            for filter_el in filter_state_dict:
                if filter_el in item[0]:
                    return True

            return False

    state_dict = dict(filter(filter_func, state_dict.items()))

    for k, v in list(state_dict.items()):
        new_k = k[len(unwanted_prefix) :]
        for old_layer_name, new_layer_name in convert_list:
            if old_layer_name in new_k:
                new_k = new_k.replace(old_layer_name, new_layer_name)

        # must do it by hand
        if ".layer_norm" in new_k and new_k.split(".layer_norm")[0][-1].isnumeric():
            new_k = new_k.replace("layer_norm", "final_layer_norm")

        state_dict[new_k] = state_dict.pop(k)

    extra_keys = set(state_dict.keys()) - set(hf_model.state_dict().keys())
    extra_keys = set(extra_keys)
    missing_keys = set(hf_model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = set({k for k in missing_keys if "final_logits_bias" not in k})
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    hf_model.load_state_dict(state_dict, strict=False)
    n_params = param_count(hf_model)

    logger.info(f"model loaded: {round(n_params/1e6,1)}M params")

    hf_model.eval()
    hf_model.to(device)
    del state_dict

    return hf_model


def load_model(pytorch_dump_folder_path, model_type):
    """
    Meta SeamlessM4T is made of 8 main components:
    - speech_encoder (#1) and speech_encoder_frontend (#2)
    - t2u_model (#3)
    - text_encoder (#4) and text_encoder_frontend (#5)
    - text_decoder (#6) [and text_decoder_frontend (#5) = equals to text_encoder_frontend]
    - final_proj (#7)
    - vocoder (#8) TODO
    """
    device = _grab_best_device()
    if model_type == "medium":
        name = "seamlessM4T_medium"
    else:
        name = "seamlessM4T_large"

    original_model = _load_original_model(device, name)

    ######### TOKENIZER

    langs = _load_langs(model_type)
    vocab_file = os.path.join(os.path.expanduser("~"), "tokenizer", model_type, "tokenizer.model")

    save_dir = os.path.join(SAVE_DIR, name)
    Path(save_dir).mkdir(exist_ok=True)

    tokenizer = SeamlessM4TTokenizer(vocab_file, language_code=langs)
    
    sanity_check_lang_id = tokenizer.lang_code_to_id["__fra__"]

    tokenizer.save_pretrained(save_dir)
    tokenizer = SeamlessM4TTokenizer.from_pretrained(save_dir)
    
    if sanity_check_lang_id != tokenizer.lang_code_to_id["__fra__"]:
        raise ValueError(f"Error in tokenizer saving/loading - __fra__ lang id is not coherent: {sanity_check_lang_id} vs {tokenizer.lang_code_to_id['__fra__']}")
    
    ######### FE
    
    fe = SeamlessM4TFeatureExtractor(language_code=langs)
    sanity_check_lang_id_fe = fe.lang_code_to_id["__fra__"]
    
    if sanity_check_lang_id != sanity_check_lang_id_fe:
        raise ValueError(f"Not coherent lang id accross FE and tokenizer: {sanity_check_lang_id} vs {sanity_check_lang_id_fe}")
    
    fe.save_pretrained(save_dir)
    fe = SeamlessM4TFeatureExtractor.from_pretrained(save_dir)
    
    if sanity_check_lang_id_fe != fe.lang_code_to_id["__fra__"]:
        raise ValueError(f"Error in FE saving/loading - __fra__ lang id is not coherent: {sanity_check_lang_id_fe} vs {fe.lang_code_to_id['__fra__']}")
    

    
    ######## Model

    # init model
    hf_config = _load_hf_config(model_type)
    hf_model = SeamlessM4TModel(hf_config)
    
    # -1. take care of vocoder
    # similarly to speech T5 must apply and remove weight norm
    hf_model.vocoder.apply_weight_norm()
    hf_model.vocoder = _convert_model(
        original_model, hf_model.vocoder, vocoder_convert_list, device, unwanted_prefix="vocoder.code_generator.", filter_state_dict="vocoder"
    )
    hf_model.vocoder.remove_weight_norm()

    # 1. take care of speech encoder
    wav2vec = hf_model.speech_encoder
    hf_model.speech_encoder = _convert_model(
        original_model, wav2vec, wav2vec_convert_list, device, unwanted_prefix="model.", filter_state_dict="speech"
    )

    # verify same number of parameters speech encoder
    count_1 = param_count(hf_model.speech_encoder)
    count_2 = param_count(original_model.model.speech_encoder_frontend) + param_count(
        original_model.model.speech_encoder
    )

    assert count_1 == count_2, f"Speech Encoder --- Count HF: {count_1} != Count Seamless: {count_2}"

    # 2. take care of t2u

    hf_model.t2u_model = _convert_model(
        original_model,
        hf_model.t2u_model,
        t2u_convert_list,
        device,
        unwanted_prefix="model.",
        filter_state_dict="t2u_model",
    )

    # verify same number of parameters t2u model
    count_1 = param_count(hf_model.t2u_model)
    count_2 = param_count(original_model.model.t2u_model)

    assert count_1 == count_2, f"T2U model --- Count HF: {count_1} != Count Seamless: {count_2}"

    # 3. take care of text encoder
    hf_model.text_encoder = _convert_model(
        original_model,
        hf_model.text_encoder,
        text_convert_list,
        device,
        unwanted_prefix="model.",
        filter_state_dict=["model.text_encoder"],
        exclude_state_dict="t2u_model",
    )

    # verify same number of parameters text_encoder
    count_1 = param_count(hf_model.text_encoder)
    count_2 = param_count(original_model.model.text_encoder) + param_count(original_model.model.text_encoder_frontend)

    assert count_1 == count_2, f"Text encoder model --- Count HF: {count_1} != Count Seamless: {count_2}"

    # 4. take care of text decoder
    hf_model.text_decoder = _convert_model(
        original_model,
        hf_model.text_decoder,
        text_convert_list,
        device,
        unwanted_prefix="model.",
        filter_state_dict=["model.text_decoder"],
        exclude_state_dict="t2u_model",
    )

    # verify same number of parameters text_decoder
    count_1 = param_count(hf_model.text_decoder)
    count_2 = param_count(original_model.model.text_decoder) + param_count(original_model.model.text_decoder_frontend)

    # with tempfile.TemporaryDirectory() as tmpdirname:
    #    hf_model.save_pretrained(tmpdirname)
    #    hf_model = SeamlessM4TModel.from_pretrained(tmpdirname)

    assert count_1 == count_2, f"Text decoder model --- Count HF: {count_1} != Count Seamless: {count_2}"

    # 5. take care of final proj
    hf_model.lm_head = _convert_model(
        original_model,
        hf_model.lm_head,
        [("final_proj.", "")],
        device,
        unwanted_prefix="model.",
        filter_state_dict=["model.final_proj"],
        exclude_state_dict="t2u_model",
    )

    # verify same number of parameters final proj
    count_1 = param_count(hf_model.lm_head)
    count_2 = param_count(original_model.model.final_proj)

    assert count_1 == count_2, f"final proj --- Count HF: {count_1} != Count Seamless: {count_2}"

    # sanity check
    print(find_tied_parameters(hf_model))

    new_model = hf_model

    count_1 = param_count(hf_model)
    count_2 = param_count(original_model)

    print(f"HF MODEL:{count_1}, ORIGINAL_MODEL: {count_2}, diff:{count_1 - count_2}")
    print(f"HF MODEL excluding embeddings:{hf_model.num_parameters(exclude_embeddings=True)}")

    del original_model

    hf_model.save_pretrained(save_dir)  # , push_to_hub=True, repo_id="ylacombe/test_seamlessM4T")
    hf_model = SeamlessM4TModel.from_pretrained(save_dir)

    input_test_text = "This is something to be translated in French"
    # dummy_speech_encoder_inputs = torch.load("/home/ubuntu/input_speech_encoder.pt")
    # attention_mask = torch.ones(input_test_text.shape[:2]).bool()
    # attention_mask[:, -1] = False
    # del attention_mask

    inputs = tokenizer([input_test_text], return_tensors="pt")

    # inputs["attention_mask"][:, -1] = 0
    set_seed(10)

    with torch.inference_mode():
        output_new_model = hf_model.generate(**inputs)

    output_text_new_model = tokenizer.decode(output_new_model[0])

    del hf_model

    original_model = _load_original_model(device)

    output_text_original_model, output_waveform_original_model, sr = original_model.predict(
        input_test_text, "T2ST", src_lang="eng", tgt_lang="fra"
    )

    output_old_model = output_waveform_original_model

    if output_text_original_model.__str__() != output_text_new_model:
        raise ValueError(
            f"Not the same text output: {output_text_original_model.__str__()} VS {output_text_new_model}"
        )

    torch.testing.assert_close(output_new_model, output_old_model)

    # output difference should come from the difference of self-attention implementation design
    if output_new_model.shape != output_old_model.shape:
        raise ValueError("initial and new outputs don't have the same shape")
    if (output_new_model - output_old_model).abs().max().item() > 1e-3:
        raise ValueError("initial and new outputs are not equal")

    #Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    #new_model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/home/yoach/m4t_weights",
        type=str,
        help="Path to the output PyTorch model.",
    )

    parser.add_argument(
        "--model_type",
        default="medium",
        type=str,
        help="Path to the output PyTorch model.",
    )

    args = parser.parse_args()

    load_model(args.pytorch_dump_folder_path, args.model_type)
