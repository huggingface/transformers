failed_tests = [
    "tests/models/altclip/test_modeling_altclip.py::AltCLIPModelIntegrationTest::test_inference_interpolate_pos_encoding",
    "tests/models/aya_vision/test_modeling_aya_vision.py::AyaVisionIntegrationTest::test_small_model_integration_batched_generate",
    "tests/models/aya_vision/test_modeling_aya_vision.py::AyaVisionIntegrationTest::test_small_model_integration_batched_generate_multi_image",
    "tests/models/chinese_clip/test_modeling_chinese_clip.py::ChineseCLIPModelIntegrationTest::test_inference_interpolate_pos_encoding",
    "tests/models/clip/test_modeling_clip.py::CLIPModelIntegrationTest::test_inference_interpolate_pos_encoding",
    "tests/models/clipseg/test_modeling_clipseg.py::CLIPSegModelIntegrationTest::test_inference_image_segmentation",
    "tests/models/clipseg/test_modeling_clipseg.py::CLIPSegModelIntegrationTest::test_inference_interpolate_pos_encoding",
    "tests/models/convnext/test_modeling_convnext.py::ConvNextModelIntegrationTest::test_inference_image_classification_head",
    "tests/models/convnextv2/test_modeling_convnextv2.py::ConvNextV2ModelIntegrationTest::test_inference_image_classification_head",
    "tests/models/cvt/test_modeling_cvt.py::CvtModelIntegrationTest::test_inference_image_classification_head",
    "tests/models/dinov2/test_modeling_dinov2.py::Dinov2ModelIntegrationTest::test_inference_no_head",
    "tests/models/dinov2_with_registers/test_modeling_dinov2_with_registers.py::Dinov2WithRegistersModelIntegrationTest::test_inference_no_head",
    "tests/models/electra/test_modeling_electra.py::ElectraModelTest::test_eager_padding_matches_padding_free_with_position_ids",
    "tests/models/flava/test_modeling_flava.py::FlavaModelIntegrationTest::test_inference",
    "tests/models/flava/test_modeling_flava.py::FlavaForPreTrainingIntegrationTest::test_inference",
    "tests/models/florence2/test_modeling_florence2.py::Florence2ForConditionalGenerationIntegrationTest::test_base_model_batching_inference_eager",
    "tests/models/florence2/test_modeling_florence2.py::Florence2ForConditionalGenerationIntegrationTest::test_large_model_batching_inference_eager",
    "tests/models/focalnet/test_modeling_focalnet.py::FocalNetModelIntegrationTest::test_inference_image_classification_head",
    "tests/models/gemma3/test_modeling_gemma3.py::Gemma3IntegrationTest::test_model_4b_batch_crops",
    "tests/models/gemma3/test_modeling_gemma3.py::Gemma3IntegrationTest::test_model_4b_crops",
    "tests/models/gemma3/test_modeling_gemma3.py::Gemma3IntegrationTest::test_model_4b_multiimage",
    "tests/models/git/test_modeling_git.py::GitModelIntegrationTest::test_forward_pass",
    "tests/models/git/test_modeling_git.py::GitModelIntegrationTest::test_inference_image_captioning",
    "tests/models/git/test_modeling_git.py::GitModelIntegrationTest::test_inference_interpolate_pos_encoding",
    "tests/models/hiera/test_modeling_hiera.py::HieraModelIntegrationTest::test_inference_for_pretraining",
    "tests/models/hiera/test_modeling_hiera.py::HieraModelIntegrationTest::test_inference_interpolate_pos_encoding",
    "tests/models/instructblip/test_modeling_instructblip.py::InstructBlipModelIntegrationTest::test_inference_flant5_xl",
    "tests/models/janus/test_modeling_janus.py::JanusIntegrationTest::test_model_generate_images",
    "tests/models/kosmos2_5/test_modeling_kosmos2_5.py::Kosmos2_5ModelIntegrationTest::test_eager",
    "tests/models/lightglue/test_modeling_lightglue.py::LightGlueModelIntegrationTest::test_inference_without_early_stop_and_keypoint_pruning",
    "tests/models/llava/test_modeling_llava.py::LlavaForConditionalGenerationIntegrationTest::test_pixtral",
    "tests/models/llava/test_modeling_llava.py::LlavaForConditionalGenerationIntegrationTest::test_pixtral_batched",
    "tests/models/mamba/test_modeling_mamba.py::MambaIntegrationTests::test_simple_generate_cuda_kernels_big_1_cpu",
    "tests/models/metaclip_2/test_modeling_metaclip_2.py::MetaClip2ModelIntegrationTest::test_inference",
    "tests/models/mlcd/test_modeling_mlcd.py::MLCDVisionModelIntegrationTest::test_inference",
    "tests/models/mllama/test_modeling_mllama.py::MllamaForCausalLMModelTest::test_eager_padding_matches_padding_free_with_position_ids",
    "tests/models/pegasus/test_modeling_pegasus.py::PegasusStandaloneDecoderModelTest::test_generate_with_static_cache",
    "tests/models/sam/test_modeling_sam.py::SamModelIntegrationTest::test_inference_mask_generation_no_point",
    "tests/models/siglip/test_modeling_siglip.py::SiglipModelIntegrationTest::test_inference",
    "tests/models/superglue/test_modeling_superglue.py::SuperGlueModelIntegrationTest::test_inference",
    "tests/models/swin/test_modeling_swin.py::SwinModelIntegrationTest::test_inference_image_classification_head",
    "tests/models/swinv2/test_modeling_swinv2.py::Swinv2ModelIntegrationTest::test_inference_image_classification_head",
    "tests/models/unispeech_sat/test_modeling_unispeech_sat.py::UniSpeechSatRobustModelTest::test_batched_inference",
    "tests/models/vilt/test_modeling_vilt.py::ViltModelIntegrationTest::test_inference_natural_language_visual_reasoning",
    "tests/models/vision_encoder_decoder/test_modeling_vision_encoder_decoder.py::TrOCRModelIntegrationTest::test_inference_printed",
    "tests/models/wav2vec2/test_modeling_wav2vec2.py::Wav2Vec2RobustModelTest::test_batched_inference",
    "tests/models/yolos/test_modeling_yolos.py::YolosModelIntegrationTest::test_inference_object_detection_head"
]

import os
for idx, failed_test in enumerate(failed_tests[:]):
    print(f"Running test {idx}: {failed_test}")
    output_dir = "captured"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = f'{output_dir}/{failed_test.replace("/", "--").replace("::", "__")}'
    os.makedirs(output_dir, exist_ok=True)
    cmd_prefix = f"PATCH_TESTING_METHODS_TO_COLLECT_OUTPUTS=yes _PATCHED_TESTING_METHODS_OUTPUT_DIR={output_dir} HF_HOME=/mnt/cache RUN_SLOW=1 python3 -m pytest -v"
    cmd = f"{cmd_prefix} {failed_test}"
    os.system(cmd)
