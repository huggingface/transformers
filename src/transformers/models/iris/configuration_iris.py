# coding=utf-8
# Copyright 2024 The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
""" IRIS model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


# from ..deprecated._archive_maps import DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP  # noqa: F401, E402


class IrisConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`IrisModel`]. It is used to
    instantiate a IRIS model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the standard
    Iris architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        state_dim (`int`, *optional*, defaults to 17):
            The state size for the RL environment
        act_dim (`int`, *optional*, defaults to 4):
            The size of the output action space
        hidden_size (`int`, *optional*, defaults to 128):
            The size of the hidden layers
        max_ep_len (`int`, *optional*, defaults to 4096):
            The maximum length of an episode in the environment
        action_tanh (`bool`, *optional*, defaults to True):
            Whether to use a tanh activation on action prediction
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`IrisModel`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_layer (`int`, *optional*, defaults to 3):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 1):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. If unset, will default to 4 times `n_embd`.
        activation_function (`str`, *optional*, defaults to `"gelu"`):
            Activation function, to be selected in the list `["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`int`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        scale_attn_by_inverse_layer_idx (`bool`, *optional*, defaults to `False`):
            Whether to additionally scale attention weights by `1 / layer_idx + 1`.
        reorder_and_upcast_attn (`bool`, *optional*, defaults to `False`):
            Whether to scale keys (K) prior to computing attention (dot-product) and upcast attention
            dot-product/softmax to float() when training with mixed precision.

    Example:

    ```python
    >>> from transformers import IrisConfig, IrisModel

    >>> # Initializing a Iris configuration
    >>> configuration = IrisConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = IrisModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    
    model_type = "iris"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "embed_dim_world_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "vocab_size": "vocab_size",
    }
      
    def __init__(
        self,
        num_actions=4,
        vocab_size=512,
        embed_dim_tokenizer=512,
        resolution=64,
        in_channels=3,
        z_channels=512,
        ch=64,
        ch_mult=[1, 1, 1, 1, 1],
        num_res_blocks=2,
        attn_resolutions=[8, 16],
        out_ch=3,
        dropout=0.0,
        use_original_obs_actor_critic=False,
        max_ram_usage='30G',
        name_train='train_dataset',
        max_num_episodes=None,
        name_test='test_dataset',
        env_train_id = None,
        env_test_id = None,
        env_train_size= 64,
        env_train_max_episode_steps= 20000,
        env_train_noop_max= 30,
        env_train_frame_skip= 4,
        env_train_done_on_life_loss= True,
        env_train_clip_reward= False,
        env_test_size= 64,
        env_test_max_episode_steps= 108000,
        env_test_noop_max= 1,
        env_test_frame_skip= 4,
        env_test_done_on_life_loss= False,
        env_test_clip_reward= False,
        tokens_per_block=17,
        max_blocks=20,
        attention='causal',
        num_layers=10,
        num_heads=4,
        embed_dim_world_model=256,
        embed_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        epochs=600,
        device='cuda:0',
        do_checkpoint=True,
        seed=0,
        resume=False, # set by resume.sh script only.
        num_envs_collect_train=1,
        stop_after_epochs_collect_train=500,
        num_episodes_to_save_collect_train=10,
        epsilon_collect_train=0.01,
        should_sample_collect_train=True,
        temperature_collect_train=1.0,
        num_steps_collect_train=200,
        num_envs_collect_test=8,
        epsilon_collect_test=0.0,
        should_sample_collect_test=True,
        temperature_collect_test=0.5,
        num_episodes_test=16,
        should_train=True,
        learning_rate=0.0001,
        batch_num_samples_tokenizer_train=128,
        grad_acc_steps_tokenizer=1,
        max_grad_norm_tokenizer=10.0,
        start_after_epochs_tokenizer_train=25,
        steps_per_epoch_tokenizer=200,
        batch_num_samples_world_model_train=32,
        grad_acc_steps_world_model=1,
        max_grad_norm_world_model=10.0,
        weight_decay=0.01,
        start_after_epochs_world_model_train=25,
        steps_per_epoch_world_model=200,
        batch_num_samples_actor_critic_train=32,
        grad_acc_steps_actor_critic=1,
        max_grad_norm_actor_critic=10.0,
        start_after_epochs_actor_critic_train=50,
        steps_per_epoch_actor_critc=200,
        burn_in_actor_critic=20,
        gamma=0.995,
        lambda_=0.95,
        entropy_weight=0.001,
        should_eval = True,
        every = 5,
        save_reconstructions_eval=True,
        
        pad_token_id=1,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs,
        
    ):
        self.num_actions=num_actions,
        self.vocab_size=vocab_size,
        self.embed_dim_tokenizer=embed_dim_tokenizer,
        self.resolution=resolution,
        self.in_channels=in_channels,
        self.z_channels=z_channels,
        self.ch=ch,
        self.ch_mult=ch_mult,
        self.num_res_blocks=num_res_blocks,
        self.attn_resolutions=attn_resolutions,
        self.out_ch=out_ch,
        self.dropout=dropout,
        self.use_original_obs_actor_critic=use_original_obs_actor_critic,
        self.max_ram_usage=max_ram_usage,
        self.name_train=name_train,
        self.max_num_episodes=max_num_episodes,
        self.name_test=name_test,
        self.env_train_id = env_train_id,
        self.env_test_id = env_test_id,
        self.env_train_size= env_train_size,
        self.env_train_max_episode_steps= env_train_max_episode_steps,
        self.env_train_noop_max= env_train_noop_max,
        self.env_train_frame_skip= env_train_frame_skip,
        self.env_train_done_on_life_loss= env_train_done_on_life_loss,
        self.env_train_clip_reward= env_train_clip_reward,
        self.env_test_size= env_test_size,
        self.env_test_max_episode_steps= env_test_max_episode_steps,
        self.env_test_noop_max= env_test_noop_max,
        self.env_test_frame_skip= env_test_frame_skip,
        self.env_test_done_on_life_loss= env_test_done_on_life_loss,
        self.env_test_clip_reward= env_test_clip_reward,
        self.tokens_per_block=tokens_per_block,
        self.max_blocks=max_blocks,
        self.attention=attention,
        self.num_layers=num_layers,
        self.num_heads=num_heads,
        self.embed_dim_world_model=embed_dim_world_model,
        self.embed_pdrop=embed_pdrop,
        self.resid_pdrop=resid_pdrop,
        self.attn_pdrop=attn_pdrop,
        self.epochs=epochs,
        self.device=device,
        self.do_checkpoint=do_checkpoint,
        self.seed=seed,
        self.sequence_length = self.max_blocks,
        self.resume=resume, # set by resume.sh script only.
        self.burn_in_actor_critic=burn_in_actor_critic,
        self.num_envs_collect_train=num_envs_collect_train,
        self.stop_after_epochs_collect_train=stop_after_epochs_collect_train,
        self.num_episodes_to_save_collect_train=num_episodes_to_save_collect_train,
        self.epsilon_collect_train=epsilon_collect_train,
        self.should_sample_collect_train=should_sample_collect_train,
        self.temperature_collect_train=temperature_collect_train,
        self.num_steps_collect_train=num_steps_collect_train,
        self.burn_in_collect_train=self.burn_in_actor_critic,
        self.num_envs_collect_test=num_envs_collect_test,
        self.num_episodes_to_save__collect_test=self.num_episodes_to_save_collect_train,
        self.epsilon_collect_train=epsilon_collect_train,
        self.should_sample_collect_test=should_sample_collect_test,
        self.temperature_collect_test=temperature_collect_test,
        self.num_episodes_test=num_episodes_test,
        self.burn_in_collect_test=self.burn_in_actor_critic,
        self.should_train=should_train,
        self.learning_rate=learning_rate,
        self.batch_num_samples_tokenizer_train=batch_num_samples_tokenizer_train,
        self.grad_acc_steps_tokenizer=grad_acc_steps_tokenizer,
        self.max_grad_norm_tokenizer=max_grad_norm_tokenizer,
        self.start_after_epochs_tokenizer_train=start_after_epochs_tokenizer_train,
        self.steps_per_epoch_tokenizer=steps_per_epoch_tokenizer,
        self.batch_num_samples_world_model_train=batch_num_samples_world_model_train,
        self.grad_acc_steps_world_model=grad_acc_steps_world_model,
        self.max_grad_norm_world_model=max_grad_norm_world_model,
        self.weight_decay=weight_decay,
        self.start_after_epochs_world_model_train=start_after_epochs_world_model_train,
        self.steps_per_epoch_world_model=steps_per_epoch_world_model,
        self.batch_num_samples_actor_critic_train=batch_num_samples_actor_critic_train,
        self.grad_acc_steps_actor_critic=grad_acc_steps_actor_critic,
        self.max_grad_norm_actor_critic=max_grad_norm_actor_critic,
        self.start_after_epochs_actor_critic_train=start_after_epochs_actor_critic_train,
        self.steps_per_epoch_actor_critc=steps_per_epoch_actor_critc,
        self.imagine_horizon_train_actor_critic = self.sequence_length,
        self.gamma=gamma,
        self.lambda_=lambda_,
        self.entropy_weight=entropy_weight,
        self.should_eval =should_eval,
        self.every = every,
        self.batch_num_samples_tokenizer_eval=self.batch_num_samples_tokenizer_train,
        self.batch_num_samples_world_model_eval=self.batch_num_samples_world_model_train,
        self.batch_num_samples_actor_critic_eval=self.batch_num_samples_actor_critic_train,
        self.save_reconstructions_eval=save_reconstructions_eval,
        self.start_after_epochs_tokenizer_eval=self.start_after_epochs_tokenizer_train,
        self.start_after_epochs_world_model_eval=self.start_after_epochs_world_model_train,
        self.start_after_epochs_actor_critic_eval=self.start_after_epochs_actor_critic_train,
        self.num_episodes_to_save__eval=self.batch_num_samples_actor_critic_train,
        self.horizon_eval_actor_critic = self.imagine_horizon_train_actor_critic,
        
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)