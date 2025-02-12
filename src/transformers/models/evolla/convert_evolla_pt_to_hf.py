# import transformers
# from transformers import EvollaModel, EvollaConfig, EvollaForProteinText2Text
# model_feature = EvollaModel.from_pretrained("/zhouxibin/workspaces/transformers/evolla-base")
# model = EvollaForProteinText2Text.from_pretrained("/zhouxibin/workspaces/transformers/evolla-base")
# exit()


state_dict_path = "/yuanfajie/ProteinQA/weights/llama3/227_mix_10m_swissprot_zk_data_protrek_12_step=310000-global_datasample=76799744-valid_loss_key=0.59092/epoch=7-step=310000-global_datasample=76799744-valid_loss_key=0.59092.ckpt/checkpoint/mp_rank_00_model_states.pt"
import torch
state_dict = torch.load(state_dict_path)
print(state_dict.keys())

from transformers import EvollaModel, EvollaConfig, EvollaForProteinText2Text
model = EvollaForProteinText2Text(EvollaConfig())

model_state_dict = model.state_dict()
ckpt_state_dict = state_dict['module']
len(model_state_dict), len(ckpt_state_dict)

mapping_dict = {
    "llm.model.model": "model.llm",
    "llm.model.lm_head": "lm_head",
    # "llm.model.model.layers": "llm.layers"
    "protein_encoder.model.esm.contact_head": None,
    "protein_encoder.model.esm": "model.protein_encoder.model",
    "protein_encoder.model.lm_head": None,
    "protein_encoder.resampler": "model.protein_encoder.sequence_compressor_resampler"
}

applied_ckpt_state_dict = {}
for ckpt_k in ckpt_state_dict.keys():
    # Check if any key in mapping_dict is a prefix of ckpt_k
    matched = False
    for mapping_key in mapping_dict.keys():
        if ckpt_k.startswith(mapping_key):
            # Use the mapping for the prefix
            model_k_start = mapping_dict[mapping_key]
            if model_k_start is None:
                print(f"Skipping: {ckpt_k}")
                matched = True
                break
            model_k = ckpt_k.replace(mapping_key, model_k_start)
            try:
                assert ckpt_state_dict[ckpt_k].shape == model_state_dict[model_k].shape, f"Shape mismatch: {ckpt_k} -> {model_k}"
            except KeyError:
                print(ckpt_k, model_k, mapping_key)
            applied_ckpt_state_dict[model_k] = ckpt_state_dict[ckpt_k]
            matched = True
            # print(f"Matched: {ckpt_k} -> {model_k}")
            break
    if not matched:
        print(f"No match found for: {ckpt_k}, shape: {ckpt_state_dict[ckpt_k].shape}")
        break  # Break loop if further keys are not processed

model.load_state_dict(state_dict=applied_ckpt_state_dict)

model.save_pretrained('/zhouxibin/workspaces/transformers/evolla-base')