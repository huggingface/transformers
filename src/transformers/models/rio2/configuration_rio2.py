# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team and the Rio2 contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""RIO-2 configuration."""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class Rio2Config(PretrainedConfig):
    r"""
    Configuration class for [`Rio2Model`].

    RIO-2 is a generative robotics policy that preserves the original
    `allenai/MolmoAct2-SO100_101` weights as much as possible while exposing two-rate runtime:

    - S2: low-frequency semantic/context refresh from the MolmoAct2 VLM path.
    - S1: high-frequency action generation, preferably using the original
      MolmoAct2 action expert when the remote-code internals expose it.

    The default mode is `weight_preserved`: the original MolmoAct2 base is the
    source of truth, small adapters are optional, and training should normally
    update only adapters/LoRA/norm/head parameters.
    """

    model_type = "rio2"
    attribute_map = {
        "hidden_size": "s1_width",
        "num_attention_heads": "s1_heads",
        "num_hidden_layers": "s1_layers",
    }

    def __init__(
        self,
        base_model_id="allenai/MolmoAct2-SO100_101",
        norm_tag="so100_so101_molmoact2",
        rio2_variant="weight_preserved",
        runtime_mode="two_rate_weight_preserved",
        state_dim=6,
        action_dim=6,
        action_horizon=30,
        state_history_len=8,
        action_history_len=8,
        # Compact token fallback path. These remain for tests and for cases
        # where the original action expert cannot be called directly.
        s2_token_count=16,
        s2_input_width=4096,
        s2_width=1024,
        s1_width=384,
        s1_layers=6,
        s1_heads=8,
        s1_dropout=0.05,
        flow_inference_steps=4,
        temporal_ensemble_enabled=True,
        temporal_ensemble_max_chunks=4,
        temporal_ensemble_decay=0.15,
        task_memory_enabled=True,
        task_memory_slots=8,
        task_memory_ema=0.97,
        task_memory_alpha=0.25,
        task_memory_max_norm=10.0,
        # Weight-preserved MolmoAct2 path.
        use_original_s2=True,
        use_original_s1=True,
        prefer_split_action_expert=True,
        fallback_to_predict_action=True,
        action_mode="continuous",
        molmoact_num_steps=10,
        s2_refresh_hz=8.0,
        max_s2_cache_age_s=0.20,
        action_clip=1.0,
        # JEPA-style S1. This keeps the original/online S1 policy weights as
        # the action generator and adds a small latent world-model side head.
        # The target action encoder is updated by EMA and is used only for the
        # self-supervised JEPA loss.
        s1_architecture="jepa_diffusion",
        enable_jepa_s1=False,
        jepa_hidden_dim=256,
        jepa_latent_dim=256,
        jepa_layers=2,
        jepa_heads=4,
        jepa_loss_weight=0.10,
        jepa_ema_decay=0.996,
        use_jepa_action_residual=False,
        jepa_action_alpha=0.0,
        s1_policy_mode="jepa_diffusion",
        enable_jepa_diffusion=True,
        diffusion_inference_steps=1,
        diffusion_loss_weight=1.0,
        consistency_loss_weight=0.50,
        flow_loss_weight=0.10,
        jepa_action_prior_weight=0.05,
        jepa_action_prior_alpha=0.25,
        jepa_condition_alpha=1.0,
        s1_sampling_noise_scale=1.0,
        enable_s1_moe=False,
        s1_moe_num_experts=10,
        s1_moe_top_k=1,
        s1_moe_expert_hidden_dim=105472,
        s1_moe_residual_scale=0.10,
        # Tiny tuning knobs.
        train_adapters_only=True,
        enable_residual_adapter=True,
        residual_alpha=0.0,
        residual_trainable=True,
        enable_s1_lora=False,
        enable_s2_lora=False,
        lora_r=8,
        lora_alpha=16,
        # Training losses for fallback/adapter path.
        smooth_loss_weight=0.02,
        action_l1_weight=0.0,
        torch_dtype="bfloat16",
        load_base_on_init=False,
        trust_remote_code=True,
        **kwargs,
    ):
        self.base_model_id = base_model_id
        self.norm_tag = norm_tag
        self.rio2_variant = rio2_variant
        self.runtime_mode = runtime_mode

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.state_history_len = state_history_len
        self.action_history_len = action_history_len

        self.s2_token_count = s2_token_count
        self.s2_input_width = s2_input_width
        self.s2_width = s2_width
        self.s1_width = s1_width
        self.s1_layers = s1_layers
        self.s1_heads = s1_heads
        self.s1_dropout = s1_dropout
        self.flow_inference_steps = flow_inference_steps
        self.temporal_ensemble_enabled = temporal_ensemble_enabled
        self.temporal_ensemble_max_chunks = temporal_ensemble_max_chunks
        self.temporal_ensemble_decay = temporal_ensemble_decay
        self.task_memory_enabled = task_memory_enabled
        self.task_memory_slots = task_memory_slots
        self.task_memory_ema = task_memory_ema
        self.task_memory_alpha = task_memory_alpha
        self.task_memory_max_norm = task_memory_max_norm

        self.use_original_s2 = use_original_s2
        self.use_original_s1 = use_original_s1
        self.prefer_split_action_expert = prefer_split_action_expert
        self.fallback_to_predict_action = fallback_to_predict_action
        self.action_mode = action_mode
        self.molmoact_num_steps = molmoact_num_steps

        self.s2_refresh_hz = s2_refresh_hz
        self.max_s2_cache_age_s = max_s2_cache_age_s
        self.action_clip = action_clip

        self.s1_architecture = s1_architecture
        self.enable_jepa_s1 = enable_jepa_s1
        self.jepa_hidden_dim = jepa_hidden_dim
        self.jepa_latent_dim = jepa_latent_dim
        self.jepa_layers = jepa_layers
        self.jepa_heads = jepa_heads
        self.jepa_loss_weight = jepa_loss_weight
        self.jepa_ema_decay = jepa_ema_decay
        self.use_jepa_action_residual = use_jepa_action_residual
        self.jepa_action_alpha = jepa_action_alpha
        self.s1_policy_mode = s1_policy_mode
        self.enable_jepa_diffusion = enable_jepa_diffusion
        self.diffusion_inference_steps = diffusion_inference_steps
        self.diffusion_loss_weight = diffusion_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.flow_loss_weight = flow_loss_weight
        self.jepa_action_prior_weight = jepa_action_prior_weight
        self.jepa_action_prior_alpha = jepa_action_prior_alpha
        self.jepa_condition_alpha = jepa_condition_alpha
        self.s1_sampling_noise_scale = s1_sampling_noise_scale
        self.enable_s1_moe = enable_s1_moe
        self.s1_moe_num_experts = s1_moe_num_experts
        self.s1_moe_top_k = s1_moe_top_k
        self.s1_moe_expert_hidden_dim = s1_moe_expert_hidden_dim
        self.s1_moe_residual_scale = s1_moe_residual_scale

        self.train_adapters_only = train_adapters_only
        self.enable_residual_adapter = enable_residual_adapter
        self.residual_alpha = residual_alpha
        self.residual_trainable = residual_trainable
        self.enable_s1_lora = enable_s1_lora
        self.enable_s2_lora = enable_s2_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        self.smooth_loss_weight = smooth_loss_weight
        self.action_l1_weight = action_l1_weight

        self.torch_dtype = torch_dtype
        self.load_base_on_init = load_base_on_init
        self.trust_remote_code = trust_remote_code

        super().__init__(**kwargs)


__all__ = ["Rio2Config"]
