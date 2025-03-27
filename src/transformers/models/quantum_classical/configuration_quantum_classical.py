# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""量子经典同构Transformer模型的配置类 (Configuration class for Quantum Classical Isomorphic Transformer Model)"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class QuantumClassicalConfig(PretrainedConfig):
    """
    量子经典同构Transformer模型的配置类
    
    此配置类用于创建量子域和经典域交互的Transformer模型，实现宇宙自参照同构优化
    
    Configuration class for Quantum Classical Isomorphic Transformer Model
    
    This configuration class is used to create a Transformer model with quantum and classical domain interaction,
    implementing universe self-referential isomorphic optimization.
    
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            词汇表大小
            Vocabulary size
        hidden_size (`int`, *optional*, defaults to 768):
            隐藏层维度
            Hidden layer dimension
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Transformer编码器层数
            Number of Transformer encoder layers
        num_attention_heads (`int`, *optional*, defaults to 12):
            注意力头数量
            Number of attention heads
        intermediate_size (`int`, *optional*, defaults to 3072):
            前馈网络中间层维度
            Intermediate layer dimension in feed-forward network
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            激活函数
            Activation function
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            隐藏层dropout概率
            Hidden layer dropout probability
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            注意力概率dropout概率
            Attention probabilities dropout probability
        max_position_embeddings (`int`, *optional*, defaults to 512):
            位置编码最大长度
            Maximum length of position embeddings
        type_vocab_size (`int`, *optional*, defaults to 2):
            token类型词汇表大小
            Token type vocabulary size
        initializer_range (`float`, *optional*, defaults to 0.02):
            初始化范围
            Initialization range
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            Layer norm的epsilon值
            Epsilon value for layer normalization
        pad_token_id (`int`, *optional*, defaults to 0):
            padding token的ID
            ID of the padding token
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            位置编码类型
            Position embedding type
        quantum_layer_alpha (`float`, *optional*, defaults to 0.5):
            量子层的权重参数alpha，控制量子域熵最小化
            Weight parameter alpha for quantum layer, controls quantum domain entropy minimization
        classical_layer_beta (`float`, *optional*, defaults to 0.5):
            经典层的权重参数beta，控制经典域熵最大化
            Weight parameter beta for classical layer, controls classical domain entropy maximization
        mrao_gamma (`float`, *optional*, defaults to 0.5):
            无限维度递归自适应算子的gamma系数，控制动态注意力强度
            Gamma coefficient for meta recursive adaptive operator, controls dynamic attention intensity
        universe_gate_init (`float`, *optional*, defaults to 0.5):
            宇宙门初始值，控制量子-经典交互初始平衡
            Universe gate initial value, controls initial balance of quantum-classical interaction
        use_quantum_attention (`bool`, *optional*, defaults to True):
            是否使用量子注意力机制
            Whether to use quantum attention mechanism
        use_classical_refinement (`bool`, *optional*, defaults to True):
            是否使用经典优化
            Whether to use classical refinement
    """
    
    model_type = "quantum_classical"
    
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        quantum_layer_alpha=0.5,
        classical_layer_beta=0.5,
        mrao_gamma=0.5,
        universe_gate_init=0.5,
        use_quantum_attention=True,
        use_classical_refinement=True,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        
        # 量子经典宇宙自参照特定参数 (Quantum-Classical Universe Self-referential specific parameters)
        self.quantum_layer_alpha = quantum_layer_alpha  # 量子域熵权重 (Quantum domain entropy weight)
        self.classical_layer_beta = classical_layer_beta  # 经典域熵权重 (Classical domain entropy weight)
        self.mrao_gamma = mrao_gamma  # 递归自适应算子系数 (Recursive adaptive operator coefficient)
        self.universe_gate_init = universe_gate_init  # 宇宙门初始值 (Universe gate initial value)
        self.use_quantum_attention = use_quantum_attention  # 启用量子注意力 (Enable quantum attention)
        self.use_classical_refinement = use_classical_refinement  # 启用经典优化 (Enable classical refinement) 