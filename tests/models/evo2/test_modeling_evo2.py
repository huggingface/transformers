# coding=utf-8
# Copyright 2025 the HuggingFace Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# This file contains the *unit* tests for the Evo2 model, based on the
# shared CausalLMModelTester utilities. Integration tests that depend on
# public Hub checkpoints or special hardware can be added later once the
# official Evo2 weights are wired to this architecture.

import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        Evo2ForCausalLM,
        Evo2Model,
    )


class Evo2ModelTester(CausalLMModelTester):
    """
    Minimal tester for Evo2 that plugs into the shared causal LM test
    harness. We just need to specify the base and LM classes; the generic
    tester will handle:
      - building a small config
      - instantiating Evo2Model / Evo2ForCausalLM
      - running forward / loss / generate / save-load tests
    """

    if is_torch_available():
        base_model_class = Evo2Model
        lm_model_class = Evo2ForCausalLM

    # If you want to tweak the tiny test config (e.g. reduce sizes),
    # you can override `prepare_config_and_inputs` or `get_config` here.


@require_torch
class Evo2ModelTest(CausalLMModelTest, unittest.TestCase):
    """
    Generic causal LM tests for Evo2.

    These tests:
      - instantiate tiny Evo2 configs
      - run forward passes
      - check loss computation
      - check generation API
      - test save / load / from_pretrained with local weights
    """

    model_tester_class = Evo2ModelTester

    # Pipelines for this model are not wired yet; skip pipeline tests.
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True
