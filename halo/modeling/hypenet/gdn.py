# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2_norm
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from .configuration_hypenet import HypeNetConfig
from .modeling_qwen3 import Qwen3Attention, apply_rotary_pos_emb

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

# https://github.com/IDSIA/recurrent-fwp/blob/master/algorithmic/layers.py#L86C1-L146C1


class GatedDeltaNet(nn.Module):
    """
    The layer implementaion for [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464).  # noqa

    Similar to Mamba2, each layer contains around 6*hidden_size*hidden_size parameters.
    Parameter alloation when use_gate=True:
        - 0.75 * hidden_size * hidden_size for the q_proj and k_proj each
        - 1.5 * hidden_size * hidden_size for the v_proj, g_proj and o_proj each
        - Others are ignorably small.
        - In total = 0.75 * 2 + 1.5 * 3 = 6 * hidden_size * hidden_size
    NOTE: num_heads * head_dim = 0.75 * hidden_size, please make sure to set the correct num_heads and head_dim.

    Parameter allocation when use_gate=False:
        - 1 * hidden_size * hidden_size for the q_proj and k_proj each
        - 2 * hidden_size * hidden_size for the v_proj and o_proj each
        - Others are ignorably small.
        - In total = 1 * 2 + 2 * 2 = 6 * hidden_size * hidden_size

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 2.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 256.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        mode (str, Optional):
            Which Gated DeltaNet kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_beta (bool, Optional):
            Whether to use beta. Default: `True`.
        use_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        layer_idx: Optional[int] = None,
        hidden_size: int = 2048,
        expand_v: float = 2,
        # head_dim: int = 256,
        key_dim: int = 128,
        val_dim: int = 128,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        activation: Optional[str] = None,
        qk_norm: bool = False,
        use_rope: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.mode = mode

        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        # self.head_dim = head_dim
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.k_dim = self.num_kv_heads * key_dim
        self.v_dim = self.num_kv_heads * val_dim
        self.q_dim = self.num_heads * key_dim
        self.layer_idx = layer_idx
        self.activation = activation
        self.qk_norm = qk_norm
        self.use_rope = use_rope
        self.silu = nn.SiLU()

        assert mode in ['chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."

        if self.qk_norm:
            self.q_norm = RMSNorm(key_dim, eps=norm_eps)
            self.k_norm = RMSNorm(key_dim, eps=norm_eps)
        self.q_proj = nn.Linear(hidden_size, self.q_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.k_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.v_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        # self.D = nn.Parameter(torch.ones(self.num_heads))
        # self.D._no_weight_decay = True
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu',
                use_fast_conv1d=False,
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu',
                use_fast_conv1d=False,
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.v_dim,
                kernel_size=conv_size,
                activation='silu',
                use_fast_conv1d=False,
            )
        # else:
        #     raise UserWarning(
        #         "ShortConvolution is crucial to the performance. "
        #         "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing."
        #     )
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.num_heads * self.val_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.val_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.val_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.num_heads * self.val_dim, hidden_size, bias=False)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        attention_mask = None
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            q, conv_state_q = self.q_conv1d(x=self.q_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_q,
                                            output_final_state=use_cache)
            k, conv_state_k = self.k_conv1d(x=self.k_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_k,
                                            output_final_state=use_cache)
            v, conv_state_v = self.v_conv1d(x=self.v_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_v,
                                            output_final_state=use_cache)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            if self.activation is not None:
                q = self.silu(q)
                k = self.silu(k)
                v = self.silu(v)

        q = rearrange(q, 'b t (h d) -> b t h d', d=self.key_dim)
        k = rearrange(k, 'b t (h d) -> b t h d', d=self.key_dim)
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.val_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            assert position_embeddings is not None
            cos, sin = position_embeddings
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            q, k = q.transpose(1, 2), k.transpose(1, 2)

        q = l2_norm(q)
        k = l2_norm(k)
        # Allow negative eigenvalues
        beta = self.b_proj(hidden_states).sigmoid() * 2
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        # Handle grouped-query, maybe we should untie the weights to go back to MHA?
        if self.num_kv_heads < self.num_heads:
            group_size = self.num_heads // self.num_kv_heads
            k = repeat(k, 'b t h d -> b t (h g) d', g=group_size)  # (B, T, nh, dh)
            v = repeat(v, 'b t h d -> b t (h g) d', g=group_size)  # (B, T, nh, dh)

        # dealing with padding
        if attention_mask is not None:
            beta = beta.mul(attention_mask[:, -beta.shape[-2]:, None])
            g = g.mul(attention_mask[:, -g.shape[-2]:, None])

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        # offsets = kwargs.get('offsets', None)
        if mode == 'chunk':
            o, recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                # offsets=offsets,
                # head_first=False
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                # offsets=offsets,
                # head_first=False
            )
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[2]
            )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o, None, past_key_values



def build_gdn_with_attn(
    attn_layer: Qwen3Attention,
    layer_idx: int,
    config: HypeNetConfig,
) -> nn.Module:
    """
    Initialize a Gated DeltaNet block using the parameters of a Qwen3Attention layer.
    We instantiate the GDN block such that the QKVO projections have the same shape,
    then copy the weights from the Qwen3Attention layer.
    """

    gdn_block = GatedDeltaNet(
        hidden_size=config.hidden_size,
        layer_idx=layer_idx,
        expand_v=1.0,
        num_heads=config.gdn_nh,
        num_kv_heads=config.gdn_nkv,
        key_dim=config.head_dim,
        val_dim=config.head_dim,
        use_short_conv=config.gdn_use_short_conv,
        use_gate=config.gdn_use_gate,
        norm_eps=config.rms_norm_eps,
        activation=config.gdn_activation,
        qk_norm=config.gdn_use_qk_norm,
        use_rope=config.gdn_use_rope,
    )

    q_proj: nn.Linear = attn_layer.q_proj
    k_proj: nn.Linear = attn_layer.k_proj
    v_proj: nn.Linear = attn_layer.v_proj
    o_proj: nn.Linear = attn_layer.o_proj
    # Note that the `.weight.shape` for a projection from d1 to d2 is (d2, d1)
    wq: Tensor = q_proj.weight  # (nh * dh, d)
    wk: Tensor = k_proj.weight  # (nkv * dh, d)
    wv: Tensor = v_proj.weight  # (nkv * dh, d)
    wo: Tensor = o_proj.weight  # (d, nh * dh)

    if config.expand_kv_proj:
        wk = wk.reshape(-1, config.head_dim, config.hidden_size)
        wv = wv.reshape(-1, config.head_dim, config.hidden_size)
        assert wk.shape[1] == wv.shape[1], wk.shape[1] == config.num_key_value_heads

        # Repeat KV projections to convert it to MHA
        target_kv_size = config.lightning_nkv * config.lightning_head_dim
        orig_kv_size = config.num_key_value_heads * config.head_dim
        expand_size = target_kv_size // orig_kv_size
        wk = wk.repeat_interleave(expand_size, dim=0)
        wv = wv.repeat_interleave(expand_size, dim=0)

        wk = wk.reshape(-1, config.hidden_size)
        wv = wv.reshape(-1, config.hidden_size)

    # ==== Create target module ====
    gdn_block.q_proj.weight.data.copy_(wq)
    gdn_block.k_proj.weight.data.copy_(wk)
    gdn_block.v_proj.weight.data.copy_(wv)
    gdn_block.o_proj.weight.data.copy_(wo)
    
    if hasattr(gdn_block, 'q_norm') and hasattr(attn_layer, 'q_norm'):
        gdn_block.q_norm.weight.data.copy_(attn_layer.q_norm.weight.data.clone())

    if hasattr(gdn_block, 'k_norm') and hasattr(attn_layer, 'k_norm'):
        gdn_block.k_norm.weight.data.copy_(attn_layer.k_norm.weight.data.clone())


    return gdn_block
