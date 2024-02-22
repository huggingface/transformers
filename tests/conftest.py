import pytest


NOT_DEVICE_TESTS = {
    "test_tokenization",
    "test_processor",
    "test_processing",
    "test_feature_extraction",
    "test_image_processing",
    "test_image_processor",
    "test_retrieval",
    "test_config",
    "test_from_pretrained_no_checkpoint",
    "test_keep_in_fp32_modules",
    "test_gradient_checkpointing_backward_compatibility",
    "test_gradient_checkpointing_enable_disable",
    "test_save_load_fast_init_from_base",
    "test_fast_init_context_manager",
    "test_fast_init_tied_embeddings",
    "test_save_load_fast_init_to_base",
    "test_torch_save_load",
    "test_initialization",
    "test_forward_signature",
    "test_model_common_attributes",
    "test_model_main_input_name",
    "test_correct_missing_keys",
    "test_tie_model_weights",
    "test_can_use_safetensors",
    "test_load_save_without_tied_weights",
    "test_tied_weights_keys",
    "test_model_weights_reload_no_missing_tied_weights",
    "test_pt_tf_model_equivalence",
    "test_mismatched_shapes_have_properly_initialized_weights",
    "test_matched_shapes_have_loaded_weights_when_some_mismatched_shapes_exist",
    "test_model_is_small",
    "test_tf_from_pt_safetensors",
    "test_flax_from_pt_safetensors",
}


def pytest_collection_modifyitems(items):
    for item in items:
        if any(test_name in item.nodeid for test_name in NOT_DEVICE_TESTS):
            item.add_marker(pytest.mark.not_device_test)
