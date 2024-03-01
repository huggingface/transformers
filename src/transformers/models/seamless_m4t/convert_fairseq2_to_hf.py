# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
from seamless_communication.models.inference.translator import Translator

from transformers import (
    SeamlessM4TConfig,
    SeamlessM4TFeatureExtractor,
    SeamlessM4TModel,
    SeamlessM4TProcessor,
    SeamlessM4TTokenizer,
)
from transformers.utils import logging


UNIT_SUPPORTED_LANGUAGES = ["__arb__", "__ben__", "__cat__", "__ces__", "__cmn__", "__cym__", "__dan__", "__deu__", "__eng__", "__est__", "__fin__", "__fra__", "__hin__", "__ind__", "__ita__", "__jpn__", "__kan__", "__kor__", "__mlt__", "__nld__", "__pes__", "__pol__", "__por__", "__ron__", "__rus__", "__slk__", "__spa__", "__swe__", "__swh__", "__tam__", "__tel__", "__tgl__", "__tha__", "__tur__", "__ukr__", "__urd__", "__uzn__", "__vie__", ]  # fmt: skip
VOCODER_SUPPORTED_LANGUAGES = ["__arb__", "__ben__", "__cat__", "__ces__", "__cmn__", "__cym__", "__dan__", "__deu__", "__eng__", "__est__", "__fin__", "__fra__", "__hin__", "__ind__", "__ita__", "__jpn__", "__kor__", "__mlt__", "__nld__", "__pes__", "__pol__", "__por__", "__ron__", "__rus__", "__slk__", "__spa__", "__swe__", "__swh__", "__tel__", "__tgl__", "__tha__", "__tur__", "__ukr__", "__urd__", "__uzn__", "__vie__",]  # fmt: skip
MEDIUM_SUPPORTED_LANGUAGES = ["ace","ace_Latn","acm","acq","aeb","afr","ajp","aka","amh","apc","arb","ars","ary","arz","asm","ast","awa","ayr","azb","azj","bak","bam","ban","bel","bem","ben","bho","bjn","bjn_Latn","bod","bos","bug","bul","cat","ceb","ces","cjk","ckb","crh","cym","dan","deu","dik","dyu","dzo","ell","eng","epo","est","eus","ewe","fao","pes","fij","fin","fon","fra","fur","fuv","gla","gle","glg","grn","guj","hat","hau","heb","hin","hne","hrv","hun","hye","ibo","ilo","ind","isl","ita","jav","jpn","kab","kac","kam","kan","kas","kas_Deva","kat","knc","knc_Latn","kaz","kbp","kea","khm","kik","kin","kir","kmb","kon","kor","kmr","lao","lvs","lij","lim","lin","lit","lmo","ltg","ltz","lua","lug","luo","lus","mag","mai","mal","mar","min","mkd","plt","mlt","mni","khk","mos","mri","zsm","mya","nld","nno","nob","npi","nso","nus","nya","oci","gaz","ory","pag","pan","pap","pol","por","prs","pbt","quy","ron","run","rus","sag","san","sat","scn","shn","sin","slk","slv","smo","sna","snd","som","sot","spa","als","srd","srp","ssw","sun","swe","swh","szl","tam","tat","tel","tgk","tgl","tha","tir","taq","taq_Tfng","tpi","tsn","tso","tuk","tum","tur","twi","tzm","uig","ukr","umb","urd","uzn","vec","vie","war","wol","xho","ydd","yor","yue","cmn","cmn_Hant","zul",]  # fmt: skip
LARGE_SUPPORTED_LANGUAGES = ["afr","amh","arb","ary","arz","asm","azj","bel","ben","bos","bul","cat","ceb","ces","ckb","cmn","cmn_Hant","cym","dan","deu","ell","eng","est","eus","fin","fra","fuv","gaz","gle","glg","guj","heb","hin","hrv","hun","hye","ibo","ind","isl","ita","jav","jpn","kan","kat","kaz","khk","khm","kir","kor","lao","lit","lug","luo","lvs","mai","mal","mar","mkd","mlt","mni","mya","nld","nno","nob","npi","nya","ory","pan","pbt","pes","pol","por","ron","rus","sat","slk","slv","sna","snd","som","spa","srp","swe","swh","tam","tel","tgk","tgl","tha","tur","ukr","urd","uzn","vie","yor","yue","zlm","zul",]  # fmt: skip


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
    ("ups", "hifi_gan.upsampler"),
    ("conv_pre", "hifi_gan.conv_pre"),
    ("resblocks", "hifi_gan.resblocks"),
    ("conv_post", "hifi_gan.conv_post"),
    ("lang", "language_embedding"),
    ("spkr", "speaker_embedding"),
    ("dict.", "unit_embedding."),
    ("dur_predictor.conv1.0", "dur_predictor.conv1"),
    ("dur_predictor.conv2.0", "dur_predictor.conv2"),
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
    ("speech_encoder.proj1", "intermediate_ffn.intermediate_dense"),
    ("speech_encoder.proj2", "intermediate_ffn.output_dense"),
    ("speech_encoder.layer_norm", "inner_layer_norm"),
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


def _load_hf_config(model_type="medium"):
    if model_type == "medium":
        kwargs = {
            "vocab_size": 256206,
            "t2u_vocab_size": 10082,
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


def load_model(save_dir, model_type, repo_id):
    """
    Meta SeamlessM4T is made of 8 main components:
    - speech_encoder (#1) and speech_encoder_frontend (#2)
    - t2u_model (#3)
    - text_encoder (#4) and text_encoder_frontend (#5)
    - text_decoder (#6) [and text_decoder_frontend (#5) = equals to text_encoder_frontend]
    - final_proj (#7)
    - vocoder (#8)
    """
    device = _grab_best_device()
    if model_type == "medium":
        name = "seamlessM4T_medium"
    else:
        name = "seamlessM4T_large"

    original_model = Translator(name, "vocoder_36langs", device, torch.float32)

    ######### TOKENIZER

    langs = MEDIUM_SUPPORTED_LANGUAGES if model_type == "medium" else LARGE_SUPPORTED_LANGUAGES
    langs = [f"__{lang}__" for lang in langs]
    vocab_file = os.path.join(os.path.expanduser("~"), "tokenizer", model_type, "tokenizer.model")

    save_dir = os.path.join(save_dir, name)
    Path(save_dir).mkdir(exist_ok=True)

    tokenizer = SeamlessM4TTokenizer(vocab_file, additional_special_tokens=langs)

    sanity_check_lang_id = tokenizer.convert_tokens_to_ids("__fra__")

    tokenizer.save_pretrained(save_dir)
    tokenizer = SeamlessM4TTokenizer.from_pretrained(save_dir)

    if sanity_check_lang_id != tokenizer.convert_tokens_to_ids("__fra__"):
        raise ValueError(
            f"Error in tokenizer saving/loading - __fra__ lang id is not coherent: {sanity_check_lang_id} vs {tokenizer.convert_tokens_to_ids('__fra__')}"
        )

    ####### get language to ids dict
    text_decoder_lang_code_to_id = {lang.replace("__", ""): tokenizer.convert_tokens_to_ids(lang) for lang in langs}
    # offset: vocoder unit vocab size + 5 (for EOS/PAD/BOS/UNK/MSK) + len(supported_languages)
    t2u_lang_code_to_id = {
        code.replace("__", ""): i + 10005 + len(UNIT_SUPPORTED_LANGUAGES)
        for i, code in enumerate(UNIT_SUPPORTED_LANGUAGES)
    }
    vocoder_lang_code_to_id = {code.replace("__", ""): i for i, code in enumerate(VOCODER_SUPPORTED_LANGUAGES)}

    ######### FE

    fe = SeamlessM4TFeatureExtractor(language_code=langs)

    fe.save_pretrained(save_dir)
    fe = SeamlessM4TFeatureExtractor.from_pretrained(save_dir)

    processor = SeamlessM4TProcessor(feature_extractor=fe, tokenizer=tokenizer)
    processor.save_pretrained(save_dir)
    processor.push_to_hub(repo_id=repo_id, create_pr=True)

    processor = SeamlessM4TProcessor.from_pretrained(save_dir)

    ######## Model

    # init model
    hf_config = _load_hf_config(model_type)
    hf_model = SeamlessM4TModel(hf_config)

    hf_model.generation_config.__setattr__("text_decoder_lang_to_code_id", text_decoder_lang_code_to_id)
    hf_model.generation_config.__setattr__("t2u_lang_code_to_id", t2u_lang_code_to_id)
    hf_model.generation_config.__setattr__("vocoder_lang_code_to_id", vocoder_lang_code_to_id)

    # -1. take care of vocoder
    # similarly to speech T5 must apply and remove weight norm
    hf_model.vocoder.apply_weight_norm()
    hf_model.vocoder = _convert_model(
        original_model,
        hf_model.vocoder,
        vocoder_convert_list,
        device,
        unwanted_prefix="vocoder.code_generator.",
        filter_state_dict="vocoder",
    )
    hf_model.vocoder.remove_weight_norm()

    # 1. take care of speech encoder
    wav2vec = hf_model.speech_encoder
    hf_model.speech_encoder = _convert_model(
        original_model, wav2vec, wav2vec_convert_list, device, unwanted_prefix="model.", filter_state_dict="speech"
    )

    # 2. take care of t2u

    hf_model.t2u_model = _convert_model(
        original_model,
        hf_model.t2u_model,
        t2u_convert_list,
        device,
        unwanted_prefix="model.",
        filter_state_dict="t2u_model",
    )

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

    # sanity check
    print(find_tied_parameters(hf_model))

    count_1 = param_count(hf_model)
    count_2 = param_count(original_model)

    print(f"HF MODEL:{count_1}, ORIGINAL_MODEL: {count_2}, diff:{count_1 - count_2}")
    print(f"HF MODEL excluding embeddings:{hf_model.num_parameters(exclude_embeddings=True)}")

    del original_model

    hf_model.generation_config._from_model_config = False
    hf_model.save_pretrained(save_dir)
    hf_model.push_to_hub(repo_id=repo_id, create_pr=True)
    hf_model = SeamlessM4TModel.from_pretrained(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters

    parser.add_argument(
        "--model_type",
        default="medium",
        type=str,
        help="Model type.",
    )

    parser.add_argument(
        "--save_dir",
        default="/home/ubuntu/weights",
        type=str,
        help="Path to the output PyTorch model.",
    )

    parser.add_argument(
        "--repo_id",
        default="facebook/hf-seamless-m4t-medium",
        type=str,
        help="Repo ID.",
    )

    args = parser.parse_args()

    load_model(args.save_dir, args.model_type, args.repo_id)
