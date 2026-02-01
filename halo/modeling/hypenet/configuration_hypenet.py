# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Qwen3 model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging


logger = logging.get_logger(__name__)


class HypeNetConfig(PretrainedConfig):
    model_type = "hypenet"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen3`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        mixer_types: list[str] = [],
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        _attn_implementation: str = 'flash_attention_2',
        # Gated DeltaNet
        gdn_use_short_conv: bool = False,
        gdn_use_gate: bool = False,
        gdn_expand_v: int = 1,
        gdn_attn_mode: str = 'chunk',
        gdn_fuse_cross_entropy: bool = False,
        gdn_activation: str | None = None,
        gdn_nh: int | None = None,
        gdn_nkv: int | None = None,
        gdn_use_qk_norm: bool = False,
        gdn_use_rope: bool = False,
        # Mamba2
        mamba2_n_groups: int = 1,
        mamba2_expand_ratio: float = 1.0,
        mamba2_conv_kernel: int = 4,
        mamba2_bias: bool = False,
        mamba2_hidden_act: str | None = None,
        # Lightning attention
        lightning_use_qk_norm: bool = False,
        lightning_use_output_gate: bool = False,
        lightning_use_output_norm: bool = False,
        lightning_use_rope: bool = True,
        lightning_rope_scaling: bool | None = None,  # true: use the rope_scaling of the teacher model.
        lightning_nh: int | None = None,
        lightning_nkv: int | None = None,
        lightning_head_dim: int | None = None,
        lightning_scale: str = '1/sqrt(d)',
        lightning_use_short_conv: bool = False,
        lightning_conv_size: int = 4,
        # Kimi Delta Attention
        kda_head_dim: int | None = None,
        kda_num_heads: int | None = None,
        kda_use_conv: bool = False,
        kda_use_qk_norm: bool = True,
        kda_use_rope: bool = False,
        # Other
        expand_kv_proj: bool = False,
        loss_fn: str = 'kl_div',
        attn_use_rope: bool = True,
        fused_ce_loss: bool = True,
        shift_labels: bool = True,
        attn_logits_scaling: None | str | float = None,
        attn_use_output_gate: bool = False,
        rand_init: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window  # we check `use_sliding_window` in the modeling code
        self.max_window_layers = max_window_layers
        self.mixer_types = mixer_types
        if len(self.mixer_types) == 0:
            # The default config is Qwen3 (full attn in every layer)
            self.mixer_types = ['attn'] * self.num_hidden_layers
        else:
            self.mixer_types = mixer_types
        assert len(self.mixer_types) == self.num_hidden_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads

        if head_dim is None:
            head_dim = self.hidden_size // self.num_attention_heads

        # For Lightning Attention
        self.head_dim = head_dim
        self.lightning_use_qk_norm = lightning_use_qk_norm
        self.lightning_use_output_norm = lightning_use_output_norm
        self.lightning_use_output_gate = lightning_use_output_gate
        self.lightning_use_rope = lightning_use_rope
        self.lightning_use_short_conv = lightning_use_short_conv
        self.lightning_conv_size = lightning_conv_size
        self.expand_kv_proj = expand_kv_proj
        self.lightning_rope_scaling = lightning_rope_scaling
        self.lightning_nh = lightning_nh if lightning_nh is not None else self.num_attention_heads
        self.lightning_nkv = lightning_nkv if lightning_nkv is not None else self.num_key_value_heads
        self.lightning_head_dim = lightning_head_dim if lightning_head_dim is not None else self.head_dim
        self.lightning_scale = lightning_scale
        self.attn_use_rope = attn_use_rope
        self.fused_ce_loss = fused_ce_loss
        self.shift_labels = shift_labels
        self.attn_logits_scaling = attn_logits_scaling
        self.attn_use_output_gate = attn_use_output_gate
        
        # Kimi Delta Attention
        self.kda_head_dim = kda_head_dim if kda_head_dim is not None else self.head_dim
        self.kda_num_heads = kda_num_heads if kda_num_heads is not None else self.num_attention_heads
        self.kda_use_conv = kda_use_conv
        self.kda_use_qk_norm = kda_use_qk_norm
        self.kda_use_rope = kda_use_rope

        # Others
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        # Gated DeltaNet (GDN)
        self.gdn_use_short_conv = gdn_use_short_conv
        self.gdn_use_gate = gdn_use_gate
        self.gdn_expand_v = gdn_expand_v
        self.gdn_attn_mode = gdn_attn_mode
        self.gdn_fuse_cross_entropy = gdn_fuse_cross_entropy
        self.gdn_activation = gdn_activation
        self.gdn_nh = gdn_nh if gdn_nh is not None else self.num_attention_heads
        self.gdn_nkv = gdn_nkv if gdn_nkv is not None else self.num_key_value_heads
        self.gdn_use_qk_norm = gdn_use_qk_norm
        self.gdn_use_rope = gdn_use_rope

        # Other
        self.loss_fn = loss_fn
        self.rand_init = rand_init

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            _attn_implementation=_attn_implementation,
            **kwargs,
        )
