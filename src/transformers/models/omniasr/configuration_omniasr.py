# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""

Example of config from OmniASR-CTC-300M model

Wav2Vec2AsrConfig(
    encoder_config=Wav2Vec2EncoderConfig(
        model_dim=1024, 
        max_seq_len=4096, 
        feature_dim=512, 
        use_fbank=False, 
        first_pass_dropout_p=0.0, 
        layer_norm_features=False, 
        feature_extractor_layer_descs=[(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)], 
        feature_extractor_bias=True, 
        feature_extractor_layer_norm_convs=True, 
        feature_grad_scale=0.1, 
        num_fbank_channels=0, 
        fbank_stride=0, 
        sample_fbank_every_k=0, 
        pos_encoder_type='conv', 
        pos_encoder_depth=1, 
        pos_conv_kernel_size=128, 
        num_pos_conv_groups=16, 
        use_conformer=False, 
        num_encoder_layers=24, 
        num_encoder_attn_heads=16, 
        ffn_inner_dim=4096, 
        dropout_p=0.0, 
        attn_dropout_p=0.0, 
        ffn_inner_dropout_p=0.1, 
        layer_drop_p=0.1, 
        norm_order=<TransformerNormOrder.PRE: 1>, 
        depthwise_conv_kernel_size=0
    ), 
    target_vocab_size=10288, 
    final_dropout_p=0.0, 
    use_masking=False, 
    temporal_mask_span_len=10, 
    max_temporal_mask_prob=0.0, 
    min_num_temporal_mask_spans=2, 
    spatial_mask_span_len=64, 
    max_spatial_mask_prob=0.0, 
    min_num_spatial_mask_spans=2
)


Example of 300m model config from OmniASR-LLM-300M v2 model

Wav2Vec2LlamaConfig(
    wav2vec2_asr_config=Wav2Vec2AsrConfig(
        encoder_config=Wav2Vec2EncoderConfig(
            model_dim=1024, 
            max_seq_len=4096, 
            feature_dim=512, 
            use_fbank=False, 
            first_pass_dropout_p=0.0, 
            layer_norm_features=False, 
            feature_extractor_layer_descs=[(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)], 
            feature_extractor_bias=True, 
            feature_extractor_layer_norm_convs=True, 
            feature_grad_scale=0.1, 
            num_fbank_channels=0, 
            fbank_stride=0, 
            sample_fbank_every_k=0, 
            pos_encoder_type='conv', 
            pos_encoder_depth=1, 
            pos_conv_kernel_size=128, 
            num_pos_conv_groups=16, 
            use_conformer=False, 
            num_encoder_layers=24, 
            num_encoder_attn_heads=16, 
            ffn_inner_dim=4096, 
            dropout_p=0.0, 
            attn_dropout_p=0.0, 
            ffn_inner_dropout_p=0.1, 
            layer_drop_p=0.1, 
            norm_order=<TransformerNormOrder.PRE: 1>, 
            depthwise_conv_kernel_size=0), 
        target_vocab_size=9812, 
        final_dropout_p=0.0, 
        use_masking=False, 
        temporal_mask_span_len=10, 
        max_temporal_mask_prob=0.0, 
        min_num_temporal_mask_spans=2, 
        spatial_mask_span_len=64, 
        max_spatial_mask_prob=0.0, 
        min_num_spatial_mask_spans=2), 
    llama_config=LLaMAConfig(
        model_dim=4096, 
        max_seq_len=8192, 
        vocab_size=10288, 
        pad_idx=1, 

        tied_embeddings=False, 
        num_layers=12, 
        num_attn_heads=8, 
        num_key_value_heads=8, 
        ffn_inner_dim=4096, 
        ffn_inner_dim_scale=0.6666666666666666, 
        ffn_inner_dim_multiplier=1.0, 
        ffn_inner_dim_multiple_of=256, 
        rope_theta=10000.0, 
        use_scaled_rope=False, 
        rope_scale=LLaMARoPEScaleConfig(
            factor=8.0, 
            frequency_factors=(1.0, 4.0), 
            original_context_length=8192),
        dropout_p=0.1, 
        init_std=None, 
        init_std_scale='layer', 
        shard_embed_dim=True), 
    beam_search_config=Wav2Vec2LlamaBeamSearchConfig(
        nbest=5, 
        length_norm=False, 
        compression_window=100, 
        compression_threshold=4.0), 
    streaming_config=Wav2Vec2LlamaStreamingConfig(
        is_streaming=False, 
        segment_secs=15.0, 
        sample_rate=16000, 
        n_context_segments=1, 
        text_tokenizer='', 
        min_audio_ms=25), 
    encoder_stacking=1, 
    frozen_encoder=1, 
    lang_embeddings_p=0.5, 
    language_column_name='lang', 
    context_text_only=False, 
    n_special_tokens=1, 
    unk_idx=3, 
    bos_idx=0, 
    eos_idx=2, 
    pad_idx=1, 
    boh_idx=None, 
    eoh_idx=None,
    model_type=<ModelType.LLM_ASR_LID: 2>, 
    n_context_examples=0)


"""


from typing import Union

from ...configuration_utils import PreTrainedConfig
from ..auto import CONFIG_MAPPING, AutoConfig


# fmt: off
DEFAULT_LANGUAGE_MAPPING = {'aae_latn': 1, 'aal_latn': 2, 'abb_latn': 3, 'abi_latn': 4, 'abk_cyrl': 5, 'abn_latn': 6, 'abp_latn': 7, 'abr_latn': 8, 'abs_latn': 9, 'aca_latn': 10, 'acd_latn': 11, 'ace_latn': 12, 'acf_latn': 13, 'ach_latn': 14, 'acm_arab': 15, 'acn_latn': 16, 'acr_latn': 17, 'acu_latn': 18, 'acw_arab': 19, 'ade_latn': 20, 'adh_latn': 21, 'adj_latn': 22, 'adx_tibt': 23, 'ady_cyrl': 24, 'aeb_arab': 25, 'aec_arab': 26, 'aeu_latn': 27, 'afb_arab': 28, 'afo_latn': 29, 'afr_latn': 30, 'agd_latn': 31, 'agg_latn': 32, 'agn_latn': 33, 'agr_latn': 34, 'agu_latn': 35, 'agx_cyrl': 36, 'aha_latn': 37, 'ahk_latn': 38, 'ahl_latn': 39, 'ahs_latn': 40, 'aia_latn': 41, 'ajg_latn': 42, 'aka_latn': 43, 'akb_latn': 44, 'ake_latn': 45, 'akp_latn': 46, 'ala_latn': 47, 'alj_latn': 48, 'aln_latn': 49, 'alo_latn': 50, 'alp_latn': 51, 'als_latn': 52, 'alt_cyrl': 53, 'alz_latn': 54, 'ame_latn': 55, 'amf_latn': 56, 'amh_ethi': 57, 'ami_latn': 58, 'amk_latn': 59, 'amu_latn': 60, 'anc_latn': 61, 'ank_latn': 62, 'ann_latn': 63, 'anp_deva': 64, 'anw_latn': 65, 'any_latn': 66, 'aom_latn': 67, 'aoz_latn': 68, 'apb_latn': 69, 'apc_arab': 70, 'apd_arab': 71, 'apr_latn': 72, 'arb_arab': 73, 'arg_latn': 74, 'arl_latn': 75, 'arq_arab': 76, 'ars_arab': 77, 'ary_arab': 78, 'arz_arab': 79, 'asa_latn': 80, 'asg_latn': 81, 'asm_beng': 82, 'ast_latn': 83, 'ata_latn': 84, 'atb_latn': 85, 'atg_latn': 86, 'ati_latn': 87, 'atq_latn': 88, 'ava_cyrl': 89, 'avn_latn': 90, 'avu_latn': 91, 'awa_deva': 92, 'awb_latn': 93, 'awo_latn': 94, 'ayl_arab': 95, 'ayo_latn': 96, 'ayp_arab': 97, 'ayr_latn': 98, 'ayz_latn': 99, 'aze_arab': 100, 'aze_cyrl': 101, 'aze_latn': 102, 'azg_latn': 103, 'azz_latn': 104, 'bag_latn': 105, 'bak_cyrl': 106, 'bam_latn': 107, 'ban_latn': 108, 'bao_latn': 109, 'bas_latn': 110, 'bav_latn': 111, 'bax_latn': 112, 'bba_latn': 113, 'bbb_latn': 114, 'bbc_latn': 115, 'bbj_latn': 116, 'bbl_geor': 117, 'bbo_latn': 118, 'bbu_latn': 119, 'bcc_arab': 120, 'bcc_latn': 121, 'bce_latn': 122, 'bci_latn': 123, 'bcl_latn': 124, 'bcs_latn': 125, 'bcw_latn': 126, 'bcy_latn': 127, 'bcz_latn': 128, 'bda_latn': 129, 'bde_latn': 130, 'bdg_latn': 131, 'bdh_latn': 132, 'bdm_latn': 133, 'bdq_latn': 134, 'bdu_latn': 135, 'bdv_orya': 136, 'beb_latn': 137, 'beh_latn': 138, 'bel_cyrl': 139, 'bem_latn': 140, 'ben_beng': 141, 'bep_latn': 142, 'bew_latn': 143, 'bex_latn': 144, 'bfa_latn': 145, 'bfd_latn': 146, 'bfo_latn': 147, 'bft_arab': 148, 'bfy_deva': 149, 'bfz_deva': 150, 'bgc_deva': 151, 'bgp_arab': 152, 'bgq_deva': 153, 'bgr_latn': 154, 'bgt_latn': 155, 'bgw_deva': 156, 'bha_deva': 157, 'bhb_deva': 158, 'bhh_cyrl': 159, 'bho_deva': 160, 'bhp_latn': 161, 'bhr_latn': 162, 'bht_deva': 163, 'bhz_latn': 164, 'bib_latn': 165, 'bim_latn': 166, 'bis_latn': 167, 'biv_latn': 168, 'bjj_deva': 169, 'bjk_latn': 170, 'bjn_latn': 171, 'bjr_latn': 172, 'bjt_latn': 173, 'bjv_latn': 174, 'bjw_latn': 175, 'bjz_latn': 176, 'bkd_latn': 177, 'bkh_latn': 178, 'bkm_latn': 179, 'bkv_latn': 180, 'bky_latn': 181, 'ble_latn': 182, 'blh_latn': 183, 'blt_latn': 184, 'blx_latn': 185, 'blz_latn': 186, 'bmm_latn': 187, 'bmq_latn': 188, 'bmr_latn': 189, 'bmu_latn': 190, 'bmv_latn': 191, 'bng_beng': 192, 'bnm_latn': 193, 'bnn_latn': 194, 'bno_latn': 195, 'bnp_latn': 196, 'bns_deva': 197, 'boa_latn': 198, 'bod_tibt': 199, 'boj_latn': 200, 'bom_latn': 201, 'bor_latn': 202, 'bos_latn': 203, 'bou_latn': 204, 'bov_latn': 205, 'box_latn': 206, 'bpr_latn': 207, 'bps_latn': 208, 'bqc_latn': 209, 'bqg_latn': 210, 'bqi_arab': 211, 'bqj_latn': 212, 'bqp_latn': 213, 'bra_deva': 214, 'bre_latn': 215, 'brh_arab': 216, 'bri_latn': 217, 'bru_latn': 218, 'brx_deva': 219, 'bsc_latn': 220, 'bsh_arab': 221, 'bsj_latn': 222, 'bsk_latn': 223, 'bsq_latn': 224, 'bss_latn': 225, 'bsy_latn': 226, 'btd_latn': 227, 'btm_latn': 228, 'bts_latn': 229, 'btt_latn': 230, 'btv_arab': 231, 'btx_latn': 232, 'bud_latn': 233, 'bug_latn': 234, 'bul_cyrl': 235, 'bum_latn': 236, 'buo_latn': 237, 'bus_latn': 238, 'bux_latn': 239, 'bvb_latn': 240, 'bvc_latn': 241, 'bvz_latn': 242, 'bwq_latn': 243, 'bwr_latn': 244, 'bwu_latn': 245, 'bxf_latn': 246, 'bxk_latn': 247, 'byc_latn': 248, 'byr_latn': 249, 'bys_latn': 250, 'byv_latn': 251, 'byx_latn': 252, 'bzh_latn': 253, 'bzi_thai': 254, 'bzj_latn': 255, 'bzw_latn': 256, 'caa_latn': 257, 'cab_latn': 258, 'cac_latn': 259, 'cak_latn': 260, 'cap_latn': 261, 'car_latn': 262, 'cas_latn': 263, 'cat_latn': 264, 'cax_latn': 265, 'cbc_latn': 266, 'cbi_latn': 267, 'cbr_latn': 268, 'cbs_latn': 269, 'cbt_latn': 270, 'cbu_latn': 271, 'cbv_latn': 272, 'cce_latn': 273, 'ccg_latn': 274, 'cco_latn': 275, 'cdj_deva': 276, 'cdo_hans': 277, 'ceb_latn': 278, 'ceg_latn': 279, 'cek_latn': 280, 'cen_latn': 281, 'ces_latn': 282, 'cfa_latn': 283, 'cfm_latn': 284, 'cgc_latn': 285, 'cgg_latn': 286, 'che_cyrl': 287, 'chf_latn': 288, 'chq_latn': 289, 'chv_cyrl': 290, 'chz_latn': 291, 'cjk_latn': 292, 'cjo_latn': 293, 'cjp_latn': 294, 'cjs_cyrl': 295, 'ckb_arab': 296, 'ckl_latn': 297, 'cko_latn': 298, 'ckr_latn': 299, 'ckt_cyrl': 300, 'cky_latn': 301, 'cla_latn': 302, 'cle_latn': 303, 'cly_latn': 304, 'cme_latn': 305, 'cmn_hans': 306, 'cmn_hant': 307, 'cmo_khmr': 308, 'cmo_latn': 309, 'cmr_latn': 310, 'cnh_latn': 311, 'cni_latn': 312, 'cnl_latn': 313, 'cnt_latn': 314, 'coe_latn': 315, 'cof_latn': 316, 'cok_latn': 317, 'con_latn': 318, 'cor_latn': 319, 'cot_latn': 320, 'cou_latn': 321, 'cpa_latn': 322, 'cpb_latn': 323, 'cpu_latn': 324, 'cpx_hans': 325, 'cpy_latn': 326, 'crh_cyrl': 327, 'crk_cans': 328, 'crk_latn': 329, 'crn_latn': 330, 'crq_latn': 331, 'crs_latn': 332, 'crt_latn': 333, 'csk_latn': 334, 'cso_latn': 335, 'ctd_latn': 336, 'cte_latn': 337, 'ctg_beng': 338, 'ctl_latn': 339, 'cto_latn': 340, 'ctu_latn': 341, 'cuc_latn': 342, 'cui_latn': 343, 'cuk_latn': 344, 'cul_latn': 345, 'cut_latn': 346, 'cux_latn': 347, 'cwa_latn': 348, 'cwe_latn': 349, 'cwt_latn': 350, 'cya_latn': 351, 'cym_latn': 352, 'daa_latn': 353, 'dag_latn': 354, 'dah_latn': 355, 'dan_latn': 356, 'dar_cyrl': 357, 'dav_latn': 358, 'dbd_latn': 359, 'dbj_latn': 360, 'dbq_latn': 361, 'dcc_arab': 362, 'ddn_latn': 363, 'ded_latn': 364, 'deg_latn': 365, 'des_latn': 366, 'deu_latn': 367, 'dga_latn': 368, 'dgh_latn': 369, 'dgi_latn': 370, 'dgk_latn': 371, 'dgo_deva': 372, 'dgr_latn': 373, 'dhi_deva': 374, 'did_latn': 375, 'dig_latn': 376, 'dik_latn': 377, 'dip_latn': 378, 'div_thaa': 379, 'dje_latn': 380, 'djk_latn': 381, 'dmk_arab': 382, 'dml_arab': 383, 'dnj_latn': 384, 'dnt_latn': 385, 'dnw_latn': 386, 'dop_latn': 387, 'dos_latn': 388, 'dru_latn': 389, 'dsb_latn': 390, 'dsh_latn': 391, 'dso_orya': 392, 'dtp_latn': 393, 'dts_latn': 394, 'dty_deva': 395, 'dua_latn': 396, 'dug_latn': 397, 'dwr_latn': 398, 'dyi_latn': 399, 'dyo_latn': 400, 'dyu_latn': 401, 'dzg_latn': 402, 'dzo_tibt': 403, 'ebu_latn': 404, 'ego_latn': 405, 'eip_latn': 406, 'eiv_latn': 407, 'eka_latn': 408, 'ekk_latn': 409, 'eko_latn': 410, 'ekr_latn': 411, 'ell_grek': 412, 'ell_grek_cypr1249': 413, 'elm_latn': 414, 'emp_latn': 415, 'enb_latn': 416, 'eng_latn': 417, 'enx_latn': 418, 'epo_latn': 419, 'ese_latn': 420, 'ess_latn': 421, 'esu_latn': 422, 'eto_latn': 423, 'ets_latn': 424, 'etu_latn': 425, 'eus_latn': 426, 'evn_cyrl': 427, 'ewe_latn': 428, 'ewo_latn': 429, 'eyo_latn': 430, 'eza_latn': 431, 'fal_latn': 432, 'fan_latn': 433, 'fao_latn': 434, 'far_latn': 435, 'fas_arab': 436, 'fat_latn': 437, 'fia_latn': 438, 'fij_latn': 439, 'fil_latn': 440, 'fin_latn': 441, 'fip_latn': 442, 'fkk_latn': 443, 'flr_latn': 444, 'fmp_latn': 445, 'fmu_deva': 446, 'fon_latn': 447, 'fra_latn': 448, 'frd_latn': 449, 'fry_latn': 450, 'fub_latn': 451, 'fuc_latn': 452, 'fue_latn': 453, 'ful_latn': 454, 'fuq_latn': 455, 'fuv_latn': 456, 'gag_cyrl': 457, 'gag_latn': 458, 'gai_latn': 459, 'gam_latn': 460, 'gau_telu': 461, 'gbi_latn': 462, 'gbk_deva': 463, 'gbm_deva': 464, 'gbo_latn': 465, 'gbr_latn': 466, 'gby_latn': 467, 'gcc_latn': 468, 'gde_latn': 469, 'gdf_latn': 470, 'geb_latn': 471, 'gej_latn': 472, 'ges_latn': 473, 'ggg_arab': 474, 'gid_latn': 475, 'gig_arab': 476, 'gil_latn': 477, 'giz_latn': 478, 'gjk_arab': 479, 'gjn_latn': 480, 'gju_arab': 481, 'gkn_latn': 482, 'gld_cyrl': 483, 'gle_latn': 484, 'glg_latn': 485, 'glk_arab': 486, 'glv_latn': 487, 'glw_latn': 488, 'gmv_latn': 489, 'gna_latn': 490, 'gnd_latn': 491, 'gng_latn': 492, 'gof_latn': 493, 'gog_latn': 494, 'gol_latn': 495, 'gom_deva': 496, 'gor_latn': 497, 'gqr_latn': 498, 'grc_grek': 499, 'gri_latn': 500, 'grn_latn': 501, 'grt_beng': 502, 'gsl_latn': 503, 'gso_latn': 504, 'gub_latn': 505, 'guc_latn': 506, 'gud_latn': 507, 'gug_latn': 508, 'guh_latn': 509, 'gui_latn': 510, 'guj_gujr': 511, 'guk_ethi': 512, 'gum_latn': 513, 'guo_latn': 514, 'guq_latn': 515, 'gur_latn': 516, 'guu_latn': 517, 'gux_latn': 518, 'guz_latn': 519, 'gvc_latn': 520, 'gvl_latn': 521, 'gwc_arab': 522, 'gwe_latn': 523, 'gwi_latn': 524, 'gwr_latn': 525, 'gwt_arab': 526, 'gym_latn': 527, 'gyr_latn': 528, 'gyz_latn': 529, 'had_latn': 530, 'hag_latn': 531, 'hah_latn': 532, 'hak_latn': 533, 'hao_latn': 534, 'hap_latn': 535, 'hat_latn': 536, 'hau_latn': 537, 'haw_latn': 538, 'hay_latn': 539, 'hbb_latn': 540, 'hch_latn': 541, 'heb_hebr': 542, 'heh_latn': 543, 'her_latn': 544, 'hia_latn': 545, 'hif_latn': 546, 'hig_latn': 547, 'hil_latn': 548, 'hin_deva': 549, 'hkk_latn': 550, 'hla_latn': 551, 'hlb_deva': 552, 'hlt_latn': 553, 'hne_deva': 554, 'hnn_latn': 555, 'hno_arab': 556, 'hns_latn': 557, 'hoc_orya': 558, 'hoy_deva': 559, 'hrv_latn': 560, 'hsb_latn': 561, 'hto_latn': 562, 'hub_latn': 563, 'hue_latn': 564, 'hui_latn': 565, 'hul_latn': 566, 'hun_latn': 567, 'hus_latn': 568, 'huu_latn': 569, 'huv_latn': 570, 'hux_latn': 571, 'hvn_latn': 572, 'hwc_latn': 573, 'hwo_latn': 574, 'hye_armn': 575, 'hyw_armn': 576, 'iba_latn': 577, 'ibb_latn': 578, 'ibo_latn': 579, 'icr_latn': 580, 'ida_latn': 581, 'idd_latn': 582, 'idu_latn': 583, 'ifa_latn': 584, 'ifb_latn': 585, 'ife_latn': 586, 'ifk_latn': 587, 'ifu_latn': 588, 'ify_latn': 589, 'igl_latn': 590, 'ign_latn': 591, 'ijc_latn': 592, 'ijn_latn': 593, 'ikk_latn': 594, 'ikw_latn': 595, 'ilb_latn': 596, 'ilo_latn': 597, 'imo_latn': 598, 'ina_latn': 599, 'inb_latn': 600, 'ind_latn': 601, 'iou_latn': 602, 'ipi_latn': 603, 'ipk_latn': 604, 'iqw_latn': 605, 'iri_latn': 606, 'irk_latn': 607, 'ish_latn': 608, 'isl_latn': 609, 'iso_latn': 610, 'ita_latn': 611, 'itl_cyrl': 612, 'its_latn': 613, 'itv_latn': 614, 'itw_latn': 615, 'itz_latn': 616, 'ixl_latn': 617, 'izr_latn': 618, 'izz_latn': 619, 'jac_latn': 620, 'jal_latn': 621, 'jam_latn': 622, 'jav_latn': 623, 'jax_latn': 624, 'jbu_latn': 625, 'jen_latn': 626, 'jic_latn': 627, 'jiv_latn': 628, 'jmc_latn': 629, 'jmd_latn': 630, 'jmx_latn': 631, 'jpn_jpan': 632, 'jqr_latn': 633, 'juk_latn': 634, 'jun_orya': 635, 'juo_latn': 636, 'juy_orya': 637, 'jvn_latn': 638, 'kaa_cyrl': 639, 'kab_latn': 640, 'kac_latn': 641, 'kai_latn': 642, 'kaj_latn': 643, 'kak_latn': 644, 'kam_latn': 645, 'kan_knda': 646, 'kao_latn': 647, 'kaq_latn': 648, 'kas_arab': 649, 'kat_geor': 650, 'kay_latn': 651, 'kaz_cyrl': 652, 'kbd_cyrl': 653, 'kbl_latn': 654, 'kbo_latn': 655, 'kbp_latn': 656, 'kbq_latn': 657, 'kbr_latn': 658, 'kbt_latn': 659, 'kby_latn': 660, 'kca_cyrl': 661, 'kcg_latn': 662, 'kcn_latn': 663, 'kcq_latn': 664, 'kdc_latn': 665, 'kde_latn': 666, 'kdh_latn': 667, 'kdi_latn': 668, 'kdj_latn': 669, 'kdl_latn': 670, 'kdn_latn': 671, 'kdt_khmr': 672, 'kea_latn': 673, 'kek_latn': 674, 'ken_latn': 675, 'keo_latn': 676, 'ker_latn': 677, 'keu_latn': 678, 'key_telu': 679, 'kez_latn': 680, 'kfb_deva': 681, 'kff_telu': 682, 'kfk_deva': 683, 'kfq_deva': 684, 'kfr_gujr': 685, 'kfw_latn': 686, 'kfx_deva': 687, 'kha_latn': 688, 'khg_tibt': 689, 'khk_cyrl': 690, 'khm_khmr': 691, 'khq_latn': 692, 'khw_arab': 693, 'kia_latn': 694, 'kij_latn': 695, 'kik_latn': 696, 'kin_latn': 697, 'kir_cyrl': 698, 'kix_latn': 699, 'kjb_latn': 700, 'kjc_latn': 701, 'kje_latn': 702, 'kjg_latn': 703, 'kjh_cyrl': 704, 'kjk_latn': 705, 'kki_latn': 706, 'kkj_latn': 707, 'kle_deva': 708, 'kln_latn': 709, 'kls_latn': 710, 'klu_latn': 711, 'klv_latn': 712, 'klw_latn': 713, 'kma_latn': 714, 'kmd_latn': 715, 'kml_latn': 716, 'kmr_arab': 717, 'kmr_cyrl': 718, 'kmr_latn': 719, 'kmu_latn': 720, 'kmy_latn': 721, 'kna_latn': 722, 'knb_latn': 723, 'knc_latn': 724, 'kne_latn': 725, 'knf_latn': 726, 'knj_latn': 727, 'knk_latn': 728, 'knn_deva': 729, 'kno_latn': 730, 'kog_latn': 731, 'kol_latn': 732, 'koo_latn': 733, 'kor_hang': 734, 'kpo_latn': 735, 'kpq_latn': 736, 'kps_latn': 737, 'kpv_cyrl': 738, 'kpy_cyrl': 739, 'kpz_latn': 740, 'kqe_latn': 741, 'kqo_latn': 742, 'kqp_latn': 743, 'kqr_latn': 744, 'kqy_ethi': 745, 'krc_cyrl': 746, 'kri_latn': 747, 'krj_latn': 748, 'krl_latn': 749, 'krr_khmr': 750, 'krs_latn': 751, 'kru_deva': 752, 'krx_latn': 753, 'ksb_latn': 754, 'ksd_latn': 755, 'ksf_latn': 756, 'ksr_latn': 757, 'kss_latn': 758, 'ksz_deva': 759, 'ktb_ethi': 760, 'ktj_latn': 761, 'kto_latn': 762, 'kua_latn': 763, 'kub_latn': 764, 'kue_latn': 765, 'kuh_latn': 766, 'kum_cyrl': 767, 'kur_arab': 768, 'kus_latn': 769, 'kvn_latn': 770, 'kvw_latn': 771, 'kvx_arab': 772, 'kwd_latn': 773, 'kwf_latn': 774, 'kwi_latn': 775, 'kwm_latn': 776, 'kxc_ethi': 777, 'kxf_latn': 778, 'kxm_thai': 779, 'kxp_arab': 780, 'kxv_orya': 781, 'kyb_latn': 782, 'kyc_latn': 783, 'kyf_latn': 784, 'kyg_latn': 785, 'kyo_latn': 786, 'kyq_latn': 787, 'kyu_kali': 788, 'kyx_latn': 789, 'kyz_latn': 790, 'kzf_latn': 791, 'kzi_latn': 792, 'lac_latn': 793, 'lag_latn': 794, 'laj_latn': 795, 'lam_latn': 796, 'lao_laoo': 797, 'las_latn': 798, 'lat_latn': 799, 'lav_latn': 800, 'law_latn': 801, 'lbj_tibt': 802, 'lbw_latn': 803, 'lcm_latn': 804, 'lcp_thai': 805, 'ldb_latn': 806, 'led_latn': 807, 'lee_latn': 808, 'lef_latn': 809, 'lem_latn': 810, 'lew_latn': 811, 'lex_latn': 812, 'lgg_latn': 813, 'lgl_latn': 814, 'lhu_latn': 815, 'lia_latn': 816, 'lid_latn': 817, 'lif_deva': 818, 'lij_latn': 819, 'lin_latn': 820, 'lip_latn': 821, 'lir_latn': 822, 'lis_lisu': 823, 'lit_latn': 824, 'lje_latn': 825, 'ljp_latn': 826, 'lkb_latn': 827, 'lke_latn': 828, 'lla_latn': 829, 'lld_latn_gherd': 830, 'lld_latn_valbadia': 831, 'llg_latn': 832, 'lln_latn': 833, 'lme_latn': 834, 'lnd_latn': 835, 'lns_latn': 836, 'lnu_latn': 837, 'loa_latn': 838, 'lob_latn': 839, 'lok_latn': 840, 'lom_latn': 841, 'lon_latn': 842, 'loq_latn': 843, 'lrk_arab': 844, 'lsi_latn': 845, 'lsm_latn': 846, 'lss_arab': 847, 'ltg_latn': 848, 'lth_latn': 849, 'lto_latn': 850, 'ltz_latn': 851, 'lua_latn': 852, 'luc_latn': 853, 'lug_latn': 854, 'luo_latn': 855, 'lus_latn': 856, 'lwg_latn': 857, 'lwo_latn': 858, 'lww_latn': 859, 'lzz_latn': 860, 'maa_latn': 861, 'mab_latn': 862, 'mad_latn': 863, 'maf_latn': 864, 'mag_deva': 865, 'mah_latn': 866, 'mai_deva': 867, 'maj_latn': 868, 'mak_latn': 869, 'mal_mlym': 870, 'mam_latn': 871, 'maq_latn': 872, 'mar_deva': 873, 'mau_latn': 874, 'maw_latn': 875, 'max_latn': 876, 'maz_latn': 877, 'mbb_latn': 878, 'mbc_latn': 879, 'mbh_latn': 880, 'mbj_latn': 881, 'mbt_latn': 882, 'mbu_latn': 883, 'mbz_latn': 884, 'mca_latn': 885, 'mcb_latn': 886, 'mcd_latn': 887, 'mcf_latn': 888, 'mco_latn': 889, 'mcp_latn': 890, 'mcq_latn': 891, 'mcu_latn': 892, 'mcx_latn': 893, 'mda_latn': 894, 'mdd_latn': 895, 'mdf_cyrl': 896, 'mdv_latn': 897, 'mdy_ethi': 898, 'med_latn': 899, 'mee_latn': 900, 'meh_latn': 901, 'mej_latn': 902, 'mek_latn': 903, 'mel_latn': 904, 'men_latn': 905, 'meq_latn': 906, 'mer_latn': 907, 'met_latn': 908, 'meu_latn': 909, 'mev_latn': 910, 'mfe_latn': 911, 'mfh_latn': 912, 'mfi_latn': 913, 'mfk_latn': 914, 'mfm_latn': 915, 'mfn_latn': 916, 'mfo_latn': 917, 'mfq_latn': 918, 'mfv_latn': 919, 'mfy_latn': 920, 'mfz_latn': 921, 'mgd_latn': 922, 'mge_latn': 923, 'mgg_latn': 924, 'mgh_latn': 925, 'mgi_latn': 926, 'mgo_latn': 927, 'mhi_latn': 928, 'mhk_latn': 929, 'mhr_cyrl': 930, 'mhu_latn': 931, 'mhx_latn': 932, 'mhy_latn': 933, 'mib_latn': 934, 'mie_latn': 935, 'mif_latn': 936, 'mig_latn': 937, 'mih_latn': 938, 'mil_latn': 939, 'mim_latn': 940, 'min_latn': 941, 'mio_latn': 942, 'mip_latn': 943, 'miq_latn': 944, 'mit_latn': 945, 'miu_latn': 946, 'miy_latn': 947, 'miz_latn': 948, 'mjl_deva': 949, 'mjv_mlym': 950, 'mkd_cyrl': 951, 'mkf_latn': 952, 'mki_arab': 953, 'mkl_latn': 954, 'mkn_latn': 955, 'mlg_latn': 956, 'mlq_latn': 957, 'mlt_latn': 958, 'mmc_latn': 959, 'mmg_latn': 960, 'mnb_latn': 961, 'mne_latn': 962, 'mnf_latn': 963, 'mni_beng': 964, 'mnk_latn': 965, 'mnw_mymr': 966, 'mnx_latn': 967, 'moa_latn': 968, 'mog_latn': 969, 'mon_cyrl': 970, 'mop_latn': 971, 'mor_latn': 972, 'mos_latn': 973, 'mox_latn': 974, 'moz_latn': 975, 'mpg_latn': 976, 'mpm_latn': 977, 'mpp_latn': 978, 'mpx_latn': 979, 'mqb_latn': 980, 'mqf_latn': 981, 'mqj_latn': 982, 'mqn_latn': 983, 'mqy_latn': 984, 'mri_latn': 985, 'mrj_cyrl': 986, 'mrr_deva': 987, 'mrt_latn': 988, 'mrw_latn': 989, 'msh_latn': 990, 'msi_latn': 991, 'msw_latn': 992, 'msy_latn': 993, 'mtd_latn': 994, 'mtj_latn': 995, 'mto_latn': 996, 'mtr_deva': 997, 'mtu_latn': 998, 'mtx_latn': 999, 'mua_latn': 1000, 'mug_latn': 1001, 'muh_latn': 1002, 'mui_latn': 1003, 'mup_deva': 1004, 'mur_latn': 1005, 'muv_mlym': 1006, 'muy_latn': 1007, 'mve_arab': 1008, 'mvp_latn': 1009, 'mvy_arab': 1010, 'mwq_latn': 1011, 'mwv_latn': 1012, 'mxb_latn': 1013, 'mxq_latn': 1014, 'mxs_latn': 1015, 'mxt_latn': 1016, 'mxu_latn': 1017, 'mxv_latn': 1018, 'mxy_latn': 1019, 'mya_mymr': 1020, 'myb_latn': 1021, 'myk_latn': 1022, 'myl_latn': 1023, 'myv_cyrl': 1024, 'myx_latn': 1025, 'myy_latn': 1026, 'mza_latn': 1027, 'mzi_latn': 1028, 'mzj_latn': 1029, 'mzk_latn': 1030, 'mzl_latn': 1031, 'mzm_latn': 1032, 'mzw_latn': 1033, 'nab_latn': 1034, 'nag_latn': 1035, 'nal_latn': 1036, 'nan_latn': 1037, 'nap_latn': 1038, 'nas_latn': 1039, 'naw_latn': 1040, 'nbh_latn': 1041, 'nca_latn': 1042, 'ncf_latn': 1043, 'nch_latn': 1044, 'ncj_latn': 1045, 'ncl_latn': 1046, 'nco_latn': 1047, 'ncu_latn': 1048, 'ncx_latn': 1049, 'ndi_latn': 1050, 'ndj_latn': 1051, 'ndo_latn': 1052, 'ndp_latn': 1053, 'ndv_latn': 1054, 'ndy_latn': 1055, 'ndz_latn': 1056, 'neb_latn': 1057, 'nep_deva': 1058, 'new_deva': 1059, 'nfa_latn': 1060, 'nfr_latn': 1061, 'nga_latn': 1062, 'ngi_latn': 1063, 'ngl_latn': 1064, 'ngp_latn': 1065, 'ngu_latn': 1066, 'nhe_latn': 1067, 'nhg_latn': 1068, 'nhi_latn': 1069, 'nhn_latn': 1070, 'nhq_latn': 1071, 'nhu_latn': 1072, 'nhw_latn': 1073, 'nhx_latn': 1074, 'nhy_latn': 1075, 'nia_latn': 1076, 'nij_latn': 1077, 'nim_latn': 1078, 'nin_latn': 1079, 'nja_latn': 1080, 'nko_latn': 1081, 'nla_latn': 1082, 'nlc_latn': 1083, 'nld_latn': 1084, 'nlg_latn': 1085, 'nlk_latn': 1086, 'nlv_latn': 1087, 'nmg_latn': 1088, 'nmz_latn': 1089, 'nnb_latn': 1090, 'nnh_latn': 1091, 'nno_latn': 1092, 'nnq_latn': 1093, 'nnw_latn': 1094, 'noa_latn': 1095, 'nob_latn': 1096, 'nod_thai': 1097, 'noe_deva': 1098, 'nog_cyrl': 1099, 'not_latn': 1100, 'npi_deva': 1101, 'npl_latn': 1102, 'npy_latn': 1103, 'nso_latn': 1104, 'nst_latn': 1105, 'nsu_latn': 1106, 'ntm_latn': 1107, 'ntr_latn': 1108, 'nuj_latn': 1109, 'nup_latn': 1110, 'nus_latn': 1111, 'nuz_latn': 1112, 'nwb_latn': 1113, 'nxq_latn': 1114, 'nya_latn': 1115, 'nyf_latn': 1116, 'nyn_latn': 1117, 'nyo_latn': 1118, 'nyu_latn': 1119, 'nyy_latn': 1120, 'nzi_latn': 1121, 'obo_latn': 1122, 'oci_latn': 1123, 'odk_arab': 1124, 'odu_latn': 1125, 'ogo_latn': 1126, 'ojb_cans': 1127, 'ojb_latn': 1128, 'oku_latn': 1129, 'old_latn': 1130, 'omw_latn': 1131, 'onb_latn': 1132, 'ood_latn': 1133, 'orc_latn': 1134, 'orm_latn': 1135, 'oru_arab': 1136, 'ory_orya': 1137, 'oss_cyrl': 1138, 'ote_latn': 1139, 'otq_latn': 1140, 'ozm_latn': 1141, 'pab_latn': 1142, 'pad_latn': 1143, 'pag_latn': 1144, 'pam_latn': 1145, 'pan_guru': 1146, 'pao_latn': 1147, 'pap_latn': 1148, 'pau_latn': 1149, 'pbb_latn': 1150, 'pbc_latn': 1151, 'pbi_latn': 1152, 'pbs_latn': 1153, 'pbt_arab': 1154, 'pbu_arab': 1155, 'pce_thai': 1156, 'pcm_latn': 1157, 'peg_orya': 1158, 'pex_latn': 1159, 'pez_latn': 1160, 'phl_arab': 1161, 'phr_arab': 1162, 'pib_latn': 1163, 'pil_latn': 1164, 'pip_latn': 1165, 'pir_latn': 1166, 'pis_latn': 1167, 'piy_latn': 1168, 'pjt_latn': 1169, 'pkb_latn': 1170, 'pko_latn': 1171, 'plk_arab': 1172, 'pls_latn': 1173, 'plt_latn': 1174, 'plw_latn': 1175, 'pmf_latn': 1176, 'pmq_latn': 1177, 'pms_latn': 1178, 'pmy_latn': 1179, 'pnb_arab': 1180, 'pne_latn': 1181, 'pny_latn': 1182, 'poc_latn': 1183, 'poe_latn': 1184, 'poh_latn': 1185, 'poi_latn': 1186, 'pol_latn': 1187, 'por_latn': 1188, 'pov_latn': 1189, 'pow_latn': 1190, 'poy_latn': 1191, 'ppk_latn': 1192, 'pps_latn': 1193, 'prf_latn': 1194, 'prk_latn': 1195, 'prq_latn': 1196, 'prt_thai': 1197, 'pse_latn': 1198, 'pss_latn': 1199, 'pst_arab': 1200, 'ptu_latn': 1201, 'pua_latn': 1202, 'pui_latn': 1203, 'pus_arab': 1204, 'pwg_latn': 1205, 'pwn_latn': 1206, 'pww_thai': 1207, 'pxm_latn': 1208, 'qub_latn': 1209, 'quc_latn': 1210, 'quf_latn': 1211, 'qug_latn': 1212, 'quh_latn': 1213, 'qul_latn': 1214, 'qum_latn': 1215, 'qup_latn': 1216, 'qur_latn': 1217, 'qus_latn': 1218, 'quv_latn': 1219, 'quw_latn': 1220, 'qux_latn': 1221, 'quy_latn': 1222, 'quz_latn': 1223, 'qva_latn': 1224, 'qvc_latn': 1225, 'qve_latn': 1226, 'qvh_latn': 1227, 'qvi_latn': 1228, 'qvj_latn': 1229, 'qvl_latn': 1230, 'qvm_latn': 1231, 'qvn_latn': 1232, 'qvo_latn': 1233, 'qvs_latn': 1234, 'qvw_latn': 1235, 'qvz_latn': 1236, 'qwa_latn': 1237, 'qwh_latn': 1238, 'qws_latn': 1239, 'qxa_latn': 1240, 'qxh_latn': 1241, 'qxl_latn': 1242, 'qxn_latn': 1243, 'qxo_latn': 1244, 'qxp_latn': 1245, 'qxr_latn': 1246, 'qxt_latn': 1247, 'qxu_latn': 1248, 'qxw_latn': 1249, 'rag_latn': 1250, 'rah_beng': 1251, 'rai_latn': 1252, 'rap_latn': 1253, 'rav_deva': 1254, 'raw_latn': 1255, 'rej_latn': 1256, 'rel_latn': 1257, 'rgu_latn': 1258, 'rhg_latn': 1259, 'rif_arab': 1260, 'rif_latn': 1261, 'ril_latn': 1262, 'rim_latn': 1263, 'rjs_deva': 1264, 'rkt_beng': 1265, 'rmc_cyrl': 1266, 'rmc_latn': 1267, 'rmo_latn': 1268, 'rmy_cyrl': 1269, 'rmy_latn': 1270, 'rng_latn': 1271, 'rnl_latn': 1272, 'rob_latn': 1273, 'rof_latn': 1274, 'roh_latn_lowe1386': 1275, 'roh_latn_surs1244': 1276, 'rol_latn': 1277, 'ron_latn': 1278, 'roo_latn': 1279, 'rop_latn': 1280, 'rro_latn': 1281, 'rth_latn': 1282, 'rub_latn': 1283, 'ruc_latn': 1284, 'ruf_latn': 1285, 'rug_latn': 1286, 'run_latn': 1287, 'rus_cyrl': 1288, 'rwm_latn': 1289, 'rwr_deva': 1290, 'sab_latn': 1291, 'sag_latn': 1292, 'sah_cyrl': 1293, 'saj_latn': 1294, 'saq_latn': 1295, 'sas_latn': 1296, 'sat_olck': 1297, 'sau_latn': 1298, 'say_latn': 1299, 'sba_latn': 1300, 'sbd_latn': 1301, 'sbl_latn': 1302, 'sbn_arab': 1303, 'sbp_latn': 1304, 'sch_latn': 1305, 'sck_deva': 1306, 'scl_arab': 1307, 'scn_latn': 1308, 'sco_latn': 1309, 'sda_latn': 1310, 'sdo_latn': 1311, 'sea_latn': 1312, 'seh_latn': 1313, 'sei_latn': 1314, 'ses_latn': 1315, 'sey_latn': 1316, 'sgb_latn': 1317, 'sgj_deva': 1318, 'sgw_ethi': 1319, 'shi_latn': 1320, 'shk_latn': 1321, 'shn_mymr': 1322, 'sho_latn': 1323, 'shp_latn': 1324, 'sid_latn': 1325, 'sig_latn': 1326, 'sil_latn': 1327, 'sin_sinh': 1328, 'sip_tibt': 1329, 'siw_latn': 1330, 'sja_latn': 1331, 'sjm_latn': 1332, 'sjp_deva': 1333, 'sjr_latn': 1334, 'skg_latn': 1335, 'skr_arab': 1336, 'sld_latn': 1337, 'slk_latn': 1338, 'slu_latn': 1339, 'slv_latn': 1340, 'sml_latn': 1341, 'smo_latn': 1342, 'sna_latn': 1343, 'snc_latn': 1344, 'snd_arab': 1345, 'sne_latn': 1346, 'snk_latn': 1347, 'snn_latn': 1348, 'snp_latn': 1349, 'snv_latn': 1350, 'snw_latn': 1351, 'sol_latn': 1352, 'som_latn': 1353, 'soy_latn': 1354, 'spa_latn': 1355, 'spp_latn': 1356, 'sps_latn': 1357, 'spy_latn': 1358, 'src_latn': 1359, 'srd_latn': 1360, 'sri_latn': 1361, 'srm_latn': 1362, 'srn_latn': 1363, 'sro_latn': 1364, 'srp_cyrl': 1365, 'srr_latn': 1366, 'srx_deva': 1367, 'ssi_arab': 1368, 'ste_latn': 1369, 'stn_latn': 1370, 'stp_latn': 1371, 'sua_latn': 1372, 'suc_latn': 1373, 'suk_latn': 1374, 'sun_latn': 1375, 'sur_latn': 1376, 'sus_latn': 1377, 'suv_latn': 1378, 'suz_deva': 1379, 'sva_geor': 1380, 'swe_latn': 1381, 'swh_latn': 1382, 'swv_deva': 1383, 'sxb_latn': 1384, 'sxn_latn': 1385, 'sya_latn': 1386, 'syl_latn': 1387, 'sza_latn': 1388, 'szy_latn': 1389, 'tac_latn': 1390, 'taj_deva': 1391, 'tam_taml': 1392, 'tan_latn': 1393, 'tao_latn': 1394, 'tap_latn': 1395, 'taq_latn': 1396, 'tar_latn': 1397, 'tat_cyrl': 1398, 'tav_latn': 1399, 'tay_latn': 1400, 'tbc_latn': 1401, 'tbf_latn': 1402, 'tbg_latn': 1403, 'tbk_latn': 1404, 'tbl_latn': 1405, 'tby_latn': 1406, 'tbz_latn': 1407, 'tca_latn': 1408, 'tcc_latn': 1409, 'tcf_latn': 1410, 'tcs_latn': 1411, 'tcy_mlym': 1412, 'tcz_latn': 1413, 'tdj_latn': 1414, 'tdn_latn': 1415, 'tdx_latn': 1416, 'ted_latn': 1417, 'tee_latn': 1418, 'tel_telu': 1419, 'tem_latn': 1420, 'teo_latn': 1421, 'ter_latn': 1422, 'tes_latn': 1423, 'tew_latn': 1424, 'tex_latn': 1425, 'tfr_latn': 1426, 'tgc_latn': 1427, 'tgj_latn': 1428, 'tgk_cyrl': 1429, 'tgl_latn': 1430, 'tgo_latn': 1431, 'tgp_latn': 1432, 'tha_thai': 1433, 'the_deva': 1434, 'thk_latn': 1435, 'thl_deva': 1436, 'thq_deva': 1437, 'thr_deva': 1438, 'thv_tfng': 1439, 'tig_ethi': 1440, 'tih_latn': 1441, 'tik_latn': 1442, 'tio_latn': 1443, 'tir_ethi': 1444, 'tkg_latn': 1445, 'tkr_latn': 1446, 'tkt_deva': 1447, 'tlb_latn': 1448, 'tli_latn': 1449, 'tlj_latn': 1450, 'tlp_latn': 1451, 'tly_latn': 1452, 'tmc_latn': 1453, 'tmf_latn': 1454, 'tna_latn': 1455, 'tng_latn': 1456, 'tnk_latn': 1457, 'tnn_latn': 1458, 'tnp_latn': 1459, 'tnr_latn': 1460, 'tnt_latn': 1461, 'tob_latn': 1462, 'toc_latn': 1463, 'toh_latn': 1464, 'tok_latn': 1465, 'tom_latn': 1466, 'top_latn': 1467, 'tos_latn': 1468, 'tpi_latn': 1469, 'tpl_latn': 1470, 'tpm_latn': 1471, 'tpp_latn': 1472, 'tpt_latn': 1473, 'tpz_latn': 1474, 'tqp_latn': 1475, 'trc_latn': 1476, 'tri_latn': 1477, 'trn_latn': 1478, 'trp_latn': 1479, 'trq_latn': 1480, 'trs_latn': 1481, 'trv_latn': 1482, 'trw_arab': 1483, 'tsn_latn': 1484, 'tso_latn': 1485, 'tsz_latn': 1486, 'ttc_latn': 1487, 'tte_latn': 1488, 'ttj_latn': 1489, 'ttq_tfng': 1490, 'ttr_latn': 1491, 'ttu_latn': 1492, 'tue_latn': 1493, 'tuf_latn': 1494, 'tui_latn': 1495, 'tuk_arab': 1496, 'tuk_latn': 1497, 'tul_latn': 1498, 'tuo_latn': 1499, 'tuq_latn': 1500, 'tur_latn': 1501, 'tuv_latn': 1502, 'tuy_latn': 1503, 'tvo_latn': 1504, 'tvu_latn': 1505, 'tvw_latn': 1506, 'twb_latn': 1507, 'twe_latn': 1508, 'twi_latn': 1509, 'twu_latn': 1510, 'txa_latn': 1511, 'txq_latn': 1512, 'txs_latn': 1513, 'txu_latn': 1514, 'txy_latn': 1515, 'tye_latn': 1516, 'tzh_latn': 1517, 'tzj_latn': 1518, 'tzo_latn': 1519, 'ubl_latn': 1520, 'ubu_latn': 1521, 'udl_latn': 1522, 'udm_cyrl': 1523, 'udu_latn': 1524, 'uig_arab': 1525, 'uig_cyrl': 1526, 'uki_orya': 1527, 'ukr_cyrl': 1528, 'ukv_latn': 1529, 'umb_latn': 1530, 'unr_orya': 1531, 'upv_latn': 1532, 'ura_latn': 1533, 'urb_latn': 1534, 'urd_arab': 1535, 'urd_deva': 1536, 'urd_latn': 1537, 'urh_latn': 1538, 'urk_thai': 1539, 'urt_latn': 1540, 'ury_latn': 1541, 'ush_arab': 1542, 'usp_latn': 1543, 'uzb_cyrl': 1544, 'uzb_latn': 1545, 'uzn_latn': 1546, 'vag_latn': 1547, 'vah_deva': 1548, 'vai_latn': 1549, 'var_latn': 1550, 'ver_latn': 1551, 'vid_latn': 1552, 'vie_latn': 1553, 'vif_latn': 1554, 'vmc_latn': 1555, 'vmj_latn': 1556, 'vmm_latn': 1557, 'vmp_latn': 1558, 'vmw_latn': 1559, 'vmy_latn': 1560, 'vmz_latn': 1561, 'vot_latn': 1562, 'vro_latn': 1563, 'vun_latn': 1564, 'vut_latn': 1565, 'wal_ethi': 1566, 'wal_latn': 1567, 'wap_latn': 1568, 'war_latn': 1569, 'waw_latn': 1570, 'way_latn': 1571, 'wba_latn': 1572, 'wbl_latn': 1573, 'wbr_deva': 1574, 'wci_latn': 1575, 'weo_latn': 1576, 'wes_latn': 1577, 'wja_latn': 1578, 'wji_latn': 1579, 'wlo_latn': 1580, 'wlx_latn': 1581, 'wmw_latn': 1582, 'wob_latn': 1583, 'wof_latn': 1584, 'wol_latn': 1585, 'wsg_telu': 1586, 'wwa_latn': 1587, 'xal_cyrl': 1588, 'xdy_latn': 1589, 'xed_latn': 1590, 'xer_latn': 1591, 'xhe_arab': 1592, 'xho_latn': 1593, 'xka_arab': 1594, 'xkl_latn': 1595, 'xmf_geor': 1596, 'xmm_latn': 1597, 'xmv_latn': 1598, 'xnj_latn': 1599, 'xnr_deva': 1600, 'xog_latn': 1601, 'xon_latn': 1602, 'xpe_latn': 1603, 'xrb_latn': 1604, 'xsb_latn': 1605, 'xsm_latn': 1606, 'xsr_deva': 1607, 'xsu_latn': 1608, 'xta_latn': 1609, 'xtd_latn': 1610, 'xte_latn': 1611, 'xti_latn': 1612, 'xtm_latn': 1613, 'xtn_latn': 1614, 'xtu_latn': 1615, 'xua_taml': 1616, 'xuo_latn': 1617, 'yaa_latn': 1618, 'yad_latn': 1619, 'yal_latn': 1620, 'yam_latn': 1621, 'yao_latn': 1622, 'yaq_latn': 1623, 'yas_latn': 1624, 'yat_latn': 1625, 'yav_latn': 1626, 'yay_latn': 1627, 'yaz_latn': 1628, 'yba_latn': 1629, 'ybb_latn': 1630, 'ycl_latn': 1631, 'ycn_latn': 1632, 'ydd_hebr': 1633, 'ydg_arab': 1634, 'yea_mlym': 1635, 'yer_latn': 1636, 'yes_latn': 1637, 'yka_latn': 1638, 'yli_latn': 1639, 'yor_latn': 1640, 'yre_latn': 1641, 'yua_latn': 1642, 'yue_hans': 1643, 'yue_hant': 1644, 'yuz_latn': 1645, 'yva_latn': 1646, 'zaa_latn': 1647, 'zab_latn': 1648, 'zac_latn': 1649, 'zad_latn': 1650, 'zae_latn': 1651, 'zai_latn': 1652, 'zam_latn': 1653, 'zao_latn': 1654, 'zaq_latn': 1655, 'zar_latn': 1656, 'zas_latn': 1657, 'zav_latn': 1658, 'zaw_latn': 1659, 'zca_latn': 1660, 'zga_latn': 1661, 'zgh_tfng': 1662, 'zim_latn': 1663, 'ziw_latn': 1664, 'zmz_latn': 1665, 'zne_latn': 1666, 'zoc_latn': 1667, 'zoh_latn': 1668, 'zor_latn': 1669, 'zos_latn': 1670, 'zpc_latn': 1671, 'zpg_latn': 1672, 'zpi_latn': 1673, 'zpl_latn': 1674, 'zpm_latn': 1675, 'zpo_latn': 1676, 'zpt_latn': 1677, 'zpu_latn': 1678, 'zpv_latn': 1679, 'zpy_latn': 1680, 'zpz_latn': 1681, 'zsm_latn': 1682, 'ztg_latn': 1683, 'ztn_latn': 1684, 'ztp_latn': 1685, 'ztq_latn': 1686, 'zts_latn': 1687, 'ztu_latn': 1688, 'zty_latn': 1689, 'zul_latn': 1690, 'zyb_latn': 1691, 'zyp_latn': 1692, 'zza_latn': 1693}
# fmt: on


class OmniASRConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OmniASRConfig`]. It is used to instantiate
    a `OmniASREncoder` model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    TODO
    - docstrings for each parameter
    - rename to Transformer convention

    """

    model_type = "omniasr"

    def __init__(
        self,
        max_seq_len=4096, 
        feature_dim=512, 
        use_fbank=False, 
        layer_norm_features=False, 
        feature_grad_scale=0.1, 
        num_fbank_channels=0, 
        fbank_stride=0, 
        sample_fbank_every_k=0, 
        pos_encoder_depth=1, 
        use_conformer=False, 
        depthwise_conv_kernel_size=0,
        # NOTE: adapted to Transformer convention
        hidden_size=1024, 
        conv_dim=[512, 512, 512, 512, 512, 512, 512],
        conv_kernel=[10, 3, 3, 3, 3, 2, 2],
        conv_stride=[5, 2, 2, 2, 2, 2, 2],
        conv_bias=True,
        layer_norm_pre=True,
        feat_extract_norm="layer", 
        num_attention_heads=16,
        num_hidden_layers=24,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        intermediate_size=4096,
        position_embeddings_type="conv",

        first_pass_dropout_p=0.0, # TODO not used
        attention_dropout=0.0,
        hidden_dropout=0.1,
        layerdrop=0.1,
        feat_proj_dropout=0.0,
        activation_dropout=0.1,

        # NOTE: added to be compatible with Wav2Vec2 modeling
        initializer_range=0.02,
        feat_extract_activation="gelu",
        layer_norm_eps=1e-5,
        hidden_act="gelu",
        add_adapter=False,
        use_intermediate_ffn_before_adapter=False,  # TODO remove?
        # TODO keep spec agument params?
        apply_spec_augment=False, 
        mask_time_length=10, 
        mask_time_prob=0.0, 
        mask_time_min_masks=2, 
        mask_feature_length=64, 
        mask_feature_prob=0.0, 
        mask_feature_min_masks=2,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        self.use_fbank = use_fbank
        self.first_pass_dropout_p = first_pass_dropout_p
        self.layer_norm_features = layer_norm_features
        self.conv_dim = conv_dim
        self.conv_stride = conv_stride
        self.conv_kernel = conv_kernel
        self.conv_bias = conv_bias
        self.feat_extract_norm = feat_extract_norm
        self.feature_grad_scale = feature_grad_scale
        self.num_fbank_channels = num_fbank_channels
        self.fbank_stride = fbank_stride
        self.sample_fbank_every_k = sample_fbank_every_k
        self.position_embeddings_type = position_embeddings_type
        self.pos_encoder_depth = pos_encoder_depth
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.use_conformer = use_conformer
        self.num_hidden_layers=num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.layerdrop = layerdrop
        # Whether layer normalization is applied at the beginning of each layer or after each layer's residuation connection: https://github.com/facebookresearch/fairseq2/blob/a510a839e007d2b036185b7b4ca76074d287c67e/src/fairseq2/models/transformer/norm_order.py#L12
        self.layer_norm_pre = layer_norm_pre
        self.depthwise_conv_kernel_size = depthwise_conv_kernel_size
        
        self.layer_norm_eps = layer_norm_eps
        self.feat_proj_dropout = feat_proj_dropout
        self.activation_dropout = activation_dropout
        self.feat_extract_activation = feat_extract_activation
        self.hidden_act = hidden_act
        self.add_adapter=add_adapter
        if use_intermediate_ffn_before_adapter and not add_adapter:
            raise ValueError("`use_intermediate_ffn_before_adapter` is `True` but `add_adapter` is `False`.")
        self.use_intermediate_ffn_before_adapter = use_intermediate_ffn_before_adapter
        
        self.initializer_range = initializer_range

        # For modular with Wav2Vec2
        self.do_stable_layer_norm = False

        # SpecAugment parameters
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_length = mask_time_length
        self.mask_time_prob = mask_time_prob
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_length = mask_feature_length
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_min_masks = mask_feature_min_masks

        super().__init__(**kwargs)

    @property
    def num_feat_extract_layers(self):
        return len(self.conv_dim)


class OmniASRCTCConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OmniASRForCTC`]. It is used to instantiate a
    OmniASR-CTC model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.
    """

    model_type = "omniasr_ctc"
    sub_configs = {"encoder_config": OmniASRConfig}

    def __init__(
        self,
        encoder_config=None,
        vocab_size=10288,
        # TODO check token ids, took from Wav2Vec2
        bos_token_id=0,
        pad_token_id=1,
        eos_token_id=2,
        unk_token_id=3,
        **kwargs,
    ):
        
        if isinstance(encoder_config, dict):
            encoder_config = OmniASRConfig(**encoder_config)
        elif encoder_config is None:
            encoder_config = OmniASRConfig()
        self.encoder_config = encoder_config

        self.vocab_size = vocab_size
        self.initializer_range = self.encoder_config.initializer_range
        self.unk_token_id = unk_token_id

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @classmethod
    def from_encoder_config(cls, encoder_config: OmniASRConfig, **kwargs):
        r"""
        Instantiate a [`OmniASRCTCConfig`] (or a derived class) from omniASR encoder model configuration.

        Returns:
            [`OmniASRCTCConfig`]: An instance of a configuration object
        """

        return cls(encoder_config=encoder_config.to_dict(), **kwargs)

    @property
    def hidden_size(self):
        return self.encoder_config.hidden_size


class OmniASRLLMConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OmniASRForConditionalGeneration`]. It is used to
    instantiate an OmniASRForConditionalGeneration model according to the specified arguments, defining the model
    architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

        
    TODO
    - docstrings for each parameter

    TODO encoder_stacking used by Zero-Shot variant
    https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/wav2vec2_llama/model.py#L1024
    """

    model_type = "omniasr_llm"
    sub_configs = {"encoder_config": OmniASRConfig, "text_config": AutoConfig}

    # TODO change to omniasr vals
    # from other repo: https://github.com/harikc456/wav2vec2_llama_hf/blob/6153f04a7d3357d49601323fc1f7f4364bce6735/convert_to_hf.py#L237
    # TODO default to 7bv2?
    """
    LLaMAConfig(model_dim=4096, max_seq_len=8192, vocab_size=10288, pad_idx=1, tied_embeddings=False, num_layers=12,            
        num_attn_heads=8, num_key_value_heads=8, ffn_inner_dim=4096, ffn_inner_dim_scale=0.6666666666666666,                        
        ffn_inner_dim_multiplier=1.0, ffn_inner_dim_multiple_of=256, rope_theta=10000.0, use_scaled_rope=False,                     
        rope_scale=LLaMARoPEScaleConfig(factor=8.0, frequency_factors=(1.0, 4.0), original_context_length=8192), dropout_p=0.1,     
        init_std=None, init_std_scale='layer', shard_embed_dim=True)  
    """
    _default_text_config_kwargs = {
        "vocab_size": 10288,
        "hidden_size": 4096,
        "num_hidden_layers": 12,
        "num_key_value_heads": 8,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-05,
        
        # "max_position_embeddings": 8192,
        "intermediate_size": 2816,
    }

    def __init__(
        self,
        encoder_config=None,
        text_config=None,
        encoder_stacking=1,
        bos_token_id=0,
        pad_token_id=1,
        eos_token_id=2,
        unk_token_id=3,
        num_special_tokens=1,
        language_mapping=None,
        language_embedding_probability=0.5,
        language_token_id=9218,
        **kwargs,
    ):
        
        if isinstance(encoder_config, dict):
            encoder_config = OmniASRConfig(**encoder_config)
        elif encoder_config is None:
            encoder_config = OmniASRConfig()
        self.encoder_config = encoder_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "llama")
            text_config = CONFIG_MAPPING[text_config["model_type"]](
                **{**self._default_text_config_kwargs, **text_config}
            )
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"](**self._default_text_config_kwargs)
        self.text_config = text_config

        self.vocab_size = text_config.vocab_size
        self.initializer_range = self.encoder_config.initializer_range
        self.unk_token_id = unk_token_id
        self.encoder_stacking = encoder_stacking
        if language_mapping is None:
            language_mapping = DEFAULT_LANGUAGE_MAPPING
        self.language_mapping = language_mapping
        self.language_embedding_probability = language_embedding_probability

        # TODO redundant? don't need num_special_tokens if we know all that are set?
        self.num_special_tokens = num_special_tokens
        self.language_token_id = language_token_id

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @property
    def num_language_embeddings(self):
        # TODO or + self.num_special_tokens? (see with zero shot model)
        return len(self.language_mapping) + 1



__all__ = ["OmniASRCTCConfig", "OmniASRLLMConfig", "OmniASRConfig"]