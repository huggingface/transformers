# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert ESM checkpoint."""


import argparse
import pathlib
from pathlib import Path
from tempfile import TemporaryDirectory

import esm as esm_module
import torch
from esm.esmfold.v1.misc import batch_encode_sequences as esmfold_encode_sequences
from esm.esmfold.v1.pretrained import esmfold_v1

from transformers.models.esm.configuration_esm import EsmConfig, EsmFoldConfig
from transformers.models.esm.modeling_esm import (
    EsmForMaskedLM,
    EsmForSequenceClassification,
    EsmIntermediate,
    EsmLayer,
    EsmOutput,
    EsmSelfAttention,
    EsmSelfOutput,
)
from transformers.models.esm.modeling_esmfold import EsmForProteinFolding
from transformers.models.esm.tokenization_esm import EsmTokenizer
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_DATA = [
    (
        "protein1",
        "MNGTEGPNFYVPFSNATGVVRSPFEYPQYYLAEPWQFSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVLGGFTSTLYTSLHGYFVFGPTGCNLEGFFATLGGEIALWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLAGWSRYIPEGLQCSCGIDYYTLKPEVNNESFVIYMFVVHFTIPMIIIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIAFLICWVPYASVAFYIFTHQGSNFGPIFMTIPAFFAKSAAIYNPVIYIMMNKQFRNCMLTTICCGKNPLGDDEASATVSKTETSQVAPA",
    ),
    ("protein2", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA"),
    ("protein3", "MKTVRQERLKSI<mask>RILERSKEPVSGAQLAEELS<mask>SRQVIVQDIAYLRSLGYN<mask>VATPRGYVLAGG"),
    ("protein4", "MKTVRQERLKSI<mask>RILERSKEPVSGAQLAEELS<mask>SRQVIVQDIAYLRSLGYN<mask>VATPRGYVLA"),
]

MODEL_MAPPING = {
    "esm1b_t33_650M_UR50S": esm_module.pretrained.esm1b_t33_650M_UR50S,
    "esm1v_t33_650M_UR90S_1": esm_module.pretrained.esm1v_t33_650M_UR90S_1,
    "esm1v_t33_650M_UR90S_2": esm_module.pretrained.esm1v_t33_650M_UR90S_2,
    "esm1v_t33_650M_UR90S_3": esm_module.pretrained.esm1v_t33_650M_UR90S_3,
    "esm1v_t33_650M_UR90S_4": esm_module.pretrained.esm1v_t33_650M_UR90S_4,
    "esm1v_t33_650M_UR90S_5": esm_module.pretrained.esm1v_t33_650M_UR90S_5,
    "esm2_t48_15B_UR50D": esm_module.pretrained.esm2_t48_15B_UR50D,
    "esm2_t36_3B_UR50D": esm_module.pretrained.esm2_t36_3B_UR50D,
    "esm2_t33_650M_UR50D": esm_module.pretrained.esm2_t33_650M_UR50D,
    "esm2_t30_150M_UR50D": esm_module.pretrained.esm2_t30_150M_UR50D,
    "esm2_t12_35M_UR50D": esm_module.pretrained.esm2_t12_35M_UR50D,
    "esm2_t6_8M_UR50D": esm_module.pretrained.esm2_t6_8M_UR50D,
    "esmfold_v1": esmfold_v1,
}

restypes = list("ARNDCQEGHILKMFPSTWYV")

restypes_with_x = restypes + ["X"]
restypes_with_extras = restypes_with_x + ["<pad>", "<mask>", "<cls>", "<sep>", "<eos>"]


def get_esmfold_tokenizer():
    with TemporaryDirectory() as tempdir:
        vocab = "\n".join(restypes_with_extras)
        vocab_file = Path(tempdir) / "vocab.txt"
        vocab_file.write_text(vocab)
        hf_tokenizer = EsmTokenizer(vocab_file=str(vocab_file))
    hf_tokenizer.pad_token_id = 0  # Overlaps with 'A' but that seems to be what they want
    return hf_tokenizer


def transfer_and_check_weights(original_module, our_module):
    status = our_module.load_state_dict(original_module.state_dict())
    if status.missing_keys:
        raise ValueError(f"Missing keys: {status.missing_keys}")
    if status.unexpected_keys:
        raise ValueError(f"Unexpected keys: {status.unexpected_keys}")


def convert_esm_checkpoint_to_pytorch(
    model: str, pytorch_dump_folder_path: str, classification_head: bool, push_to_repo: str, auth_token: str
):
    """
    Copy/paste/tweak esm's weights to our BERT structure.
    """
    if model.startswith("esmfold"):
        esm = MODEL_MAPPING[model]()
    else:
        esm, alphabet = MODEL_MAPPING[model]()
    esm.eval()  # disable dropout

    if model.startswith("esmfold"):
        embed_dim = esm.esm.embed_dim
        num_layers = esm.esm.num_layers
        num_attention_heads = esm.esm.attention_heads
        intermediate_size = 4 * embed_dim
        token_dropout = esm.esm.token_dropout
        emb_layer_norm_before = False  # This code path does not exist in ESM-2
        position_embedding_type = "rotary"
        is_folding_model = True
        esmfold_config = EsmFoldConfig()
        for key, val in esm.cfg.items():
            if hasattr(esmfold_config, key) and key != "trunk":
                setattr(esmfold_config, key, val)
        for key, val in esm.cfg.trunk.items():
            if hasattr(esmfold_config.trunk, key) and key != "structure_module":
                setattr(esmfold_config.trunk, key, val)
        for key, val in esm.cfg.trunk.structure_module.items():
            if hasattr(esmfold_config.trunk.structure_module, key):
                setattr(esmfold_config.trunk.structure_module, key, val)
    elif hasattr(esm, "args"):
        # Indicates an ESM-1b or ESM-1v model
        embed_dim = esm.args.embed_dim
        num_layers = esm.args.layers
        num_attention_heads = esm.args.attention_heads
        intermediate_size = esm.args.ffn_embed_dim
        token_dropout = esm.args.token_dropout
        emb_layer_norm_before = True if esm.emb_layer_norm_before else False
        position_embedding_type = "absolute"
        is_folding_model = False
        esmfold_config = None
    else:
        # Indicates an ESM-2 model
        embed_dim = esm.embed_dim
        num_layers = esm.num_layers
        num_attention_heads = esm.attention_heads
        intermediate_size = 4 * embed_dim  # This is hardcoded in ESM-2
        token_dropout = esm.token_dropout
        emb_layer_norm_before = False  # This code path does not exist in ESM-2
        position_embedding_type = "rotary"
        is_folding_model = False
        esmfold_config = None

    if is_folding_model:
        alphabet = esm.esm.alphabet
    vocab_list = tuple(alphabet.all_toks)
    mask_token_id = alphabet.mask_idx
    pad_token_id = alphabet.padding_idx

    if is_folding_model:
        original_esm_model = esm.esm
    else:
        original_esm_model = esm

    config = EsmConfig(
        vocab_size=original_esm_model.embed_tokens.num_embeddings,
        mask_token_id=mask_token_id,
        hidden_size=embed_dim,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=1026,
        layer_norm_eps=1e-5,  # PyTorch default used in fairseq
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        pad_token_id=pad_token_id,
        emb_layer_norm_before=emb_layer_norm_before,
        token_dropout=token_dropout,
        position_embedding_type=position_embedding_type,
        is_folding_model=is_folding_model,
        esmfold_config=esmfold_config,
        vocab_list=vocab_list,
    )
    if classification_head:
        config.num_labels = esm.classification_heads["mnli"].out_proj.weight.shape[0]
    print("Our ESM config:", config)

    if model.startswith("esmfold"):
        model_class = EsmForProteinFolding
    elif classification_head:
        model_class = EsmForSequenceClassification
    else:
        model_class = EsmForMaskedLM
    model = model_class(config)
    model.eval()

    # Now let's copy all the weights.
    # Embeddings
    model.esm.embeddings.word_embeddings.weight = original_esm_model.embed_tokens.weight
    if position_embedding_type == "absolute":
        model.esm.embeddings.position_embeddings.weight = original_esm_model.embed_positions.weight

    if config.emb_layer_norm_before:
        model.esm.embeddings.layer_norm.weight = original_esm_model.emb_layer_norm_before.weight
        model.esm.embeddings.layer_norm.bias = original_esm_model.emb_layer_norm_before.bias

    model.esm.encoder.emb_layer_norm_after.weight = original_esm_model.emb_layer_norm_after.weight
    model.esm.encoder.emb_layer_norm_after.bias = original_esm_model.emb_layer_norm_after.bias

    for i in range(config.num_hidden_layers):
        # Encoder: start of layer
        layer: EsmLayer = model.esm.encoder.layer[i]
        # esm_layer: TransformerSentenceEncoderLayer = original_esm_model.layers[i]
        esm_layer = original_esm_model.layers[i]

        # self attention
        self_attn: EsmSelfAttention = layer.attention.self
        assert (
            esm_layer.self_attn.k_proj.weight.data.shape
            == esm_layer.self_attn.q_proj.weight.data.shape
            == esm_layer.self_attn.v_proj.weight.data.shape
            == torch.Size((config.hidden_size, config.hidden_size))
        )

        self_attn.query.weight.data = esm_layer.self_attn.q_proj.weight
        self_attn.query.bias.data = esm_layer.self_attn.q_proj.bias
        self_attn.key.weight.data = esm_layer.self_attn.k_proj.weight
        self_attn.key.bias.data = esm_layer.self_attn.k_proj.bias
        self_attn.value.weight.data = esm_layer.self_attn.v_proj.weight
        self_attn.value.bias.data = esm_layer.self_attn.v_proj.bias

        if getattr(esm_layer.self_attn, "rot_emb", None) is not None:
            # Matt: Although inv_freq is not a trainable weight, it is computed at model init and cached.
            # During the training of ESM-2 the model was converted to float16 precision, which also converts
            # the inv_freq tensor, and the loss of precision remains even if the model is loaded later as float32.
            # If we recompute inv_freq without this loss of precision then we will get subtly different rotary
            # embeddings, which are enough to cause significant discrepancies in model outputs. To avoid this,
            # we make sure the new model copies the data from the old inv_freq.
            self_attn.rotary_embeddings.inv_freq.data = esm_layer.self_attn.rot_emb.inv_freq

        # LayerNorm changes for pre-activation
        layer.attention.LayerNorm.weight = esm_layer.self_attn_layer_norm.weight
        layer.attention.LayerNorm.bias = esm_layer.self_attn_layer_norm.bias
        layer.LayerNorm.weight = esm_layer.final_layer_norm.weight
        layer.LayerNorm.bias = esm_layer.final_layer_norm.bias

        # self-attention output
        self_output: EsmSelfOutput = layer.attention.output
        assert self_output.dense.weight.shape == esm_layer.self_attn.out_proj.weight.shape
        self_output.dense.weight = esm_layer.self_attn.out_proj.weight
        self_output.dense.bias = esm_layer.self_attn.out_proj.bias

        # intermediate
        intermediate: EsmIntermediate = layer.intermediate
        assert intermediate.dense.weight.shape == esm_layer.fc1.weight.shape
        intermediate.dense.weight = esm_layer.fc1.weight
        intermediate.dense.bias = esm_layer.fc1.bias

        # output
        bert_output: EsmOutput = layer.output
        assert bert_output.dense.weight.shape == esm_layer.fc2.weight.shape
        bert_output.dense.weight = esm_layer.fc2.weight
        bert_output.dense.bias = esm_layer.fc2.bias
        # end of layer

    if is_folding_model:
        model.esm_s_combine.data = esm.esm_s_combine.data
        model.af2_to_esm.data = esm.af2_to_esm.data
        transfer_and_check_weights(esm.embedding, model.embedding)
        transfer_and_check_weights(esm.esm_s_mlp, model.esm_s_mlp)
        transfer_and_check_weights(esm.trunk, model.trunk)
        transfer_and_check_weights(esm.distogram_head, model.distogram_head)
        transfer_and_check_weights(esm.ptm_head, model.ptm_head)
        transfer_and_check_weights(esm.lm_head, model.lm_head)
        transfer_and_check_weights(esm.lddt_head, model.lddt_head)

    elif classification_head:
        model.classifier.dense.weight = esm.esm.classification_heads["mnli"].dense.weight
        model.classifier.dense.bias = esm.classification_heads["mnli"].dense.bias
        model.classifier.out_proj.weight = esm.classification_heads["mnli"].out_proj.weight
        model.classifier.out_proj.bias = esm.classification_heads["mnli"].out_proj.bias
    else:
        # LM Head
        model.lm_head.dense.weight = esm.lm_head.dense.weight
        model.lm_head.dense.bias = esm.lm_head.dense.bias
        model.lm_head.layer_norm.weight = esm.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = esm.lm_head.layer_norm.bias
        model.lm_head.decoder.weight = esm.lm_head.weight
        model.lm_head.bias = esm.lm_head.bias

    # Contact prediction head
    transfer_and_check_weights(esm.contact_head, model.esm.contact_head)

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    if is_folding_model:
        # Folding models aren't trained on masked inputs and don't like mask tokens.
        sample_data = SAMPLE_DATA[:2]
    else:
        sample_data = SAMPLE_DATA

    if is_folding_model:
        hf_tokenizer = get_esmfold_tokenizer()
        hf_tokens = hf_tokenizer(
            [row[1] for row in sample_data], return_tensors="pt", padding=True, add_special_tokens=False
        )
        esmfold_aas, esmfold_mask, _, _, _ = esmfold_encode_sequences([row[1] for row in sample_data])
        success = torch.all(hf_tokens["input_ids"] == esmfold_aas) and torch.all(
            hf_tokens["attention_mask"] == esmfold_mask
        )
    else:
        # Let's check that we get the same results.
        batch_converter = alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(sample_data)
        # Prepare tokenizer and make sure it matches
        with TemporaryDirectory() as tempdir:
            vocab = "\n".join(alphabet.all_toks)
            vocab_file = Path(tempdir) / "vocab.txt"
            vocab_file.write_text(vocab)
            hf_tokenizer = EsmTokenizer(vocab_file=str(vocab_file))

        hf_tokens = hf_tokenizer([row[1] for row in sample_data], return_tensors="pt", padding=True)
        success = torch.all(hf_tokens["input_ids"] == batch_tokens)

    print("Do both models tokenizers output the same tokens?", "🔥" if success else "💩")
    if not success:
        raise Exception("Tokenization does not match!")

    with torch.no_grad():
        if is_folding_model:
            # Let's test the model in parts
            # ESMFold always converts the ESM stem to float16, which requires float16 ops
            # that don't exist on CPU. Therefore, to test it we need to run it on GPU. However,
            # ESMFold is what we in the community call a "big boy" and so we desperately avoid putting both the
            # original and the converted model on the GPU at the same time.
            their_output = esm.cuda().infer([row[1] for row in sample_data])
            our_output = model.cuda()(
                input_ids=hf_tokens["input_ids"].cuda(), attention_mask=hf_tokens["attention_mask"].cuda()
            )
        else:
            our_output = model(**hf_tokens, output_hidden_states=True)
            our_output = our_output["logits"]
            if classification_head:
                their_output = esm.model.classification_heads["mnli"](esm.extract_features(batch_tokens))
            else:
                their_output = esm(hf_tokens["input_ids"], repr_layers=list(range(999)))
                their_output = their_output["logits"]

        if is_folding_model:
            max_absolute_diff = torch.max(torch.abs(our_output["positions"] - their_output["positions"])).item()
            success = torch.allclose(our_output["positions"], their_output["positions"], atol=1e-5)
        else:
            max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
            success = torch.allclose(our_output, their_output, atol=1e-5)

        print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-5
        print("Do both models output the same tensors?", "🔥" if success else "💩")

        if not success:
            raise Exception("Something went wRoNg")

        if not is_folding_model:
            # Let's check contact prediction too
            our_output = model.predict_contacts(hf_tokens["input_ids"], hf_tokens["attention_mask"])
            their_output = esm.predict_contacts(hf_tokens["input_ids"])
            max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
            success = torch.allclose(our_output, their_output, atol=1e-5)

            print("Contact prediction testing:")
            print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-5
            print("Do both models output the same tensors?", "🔥" if success else "💩")

            if not success:
                raise Exception("Something went wRoNg")

        pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
        print(f"Saving model to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)

        del esm  # Free up some memory before continuing

    print(f"Saving tokenizer to {pytorch_dump_folder_path}")
    hf_tokenizer.save_pretrained(pytorch_dump_folder_path)

    if push_to_repo:
        model.push_to_hub(repo_id=push_to_repo, use_auth_token=auth_token)
        hf_tokenizer.push_to_hub(repo_id=push_to_repo, use_auth_token=auth_token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--pytorch_dump_folder_path", type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    parser.add_argument("--model", default=None, type=str, required=True, help="Name of model to convert.")
    parser.add_argument("--push_to_repo", type=str, help="Repo to upload to (including username!).")
    parser.add_argument("--auth_token", type=str, help="HuggingFace auth token.")
    args = parser.parse_args()
    convert_esm_checkpoint_to_pytorch(
        args.model, args.pytorch_dump_folder_path, args.classification_head, args.push_to_repo, args.auth_token
    )
