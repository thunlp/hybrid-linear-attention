from typing import Optional, Tuple
from torch import nn
import torch
from einops import rearrange, repeat
try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
    from fla.ops.utils.index import prepare_cu_seqlens_from_mask, prepare_lens_from_mask
    from fla.utils import tensor_cache
except ImportError:
    raise ImportError("Plese run `pip install -U fla-core`")
from .configuration_hypenet import HypeNetConfig
from .cache import HybridCache
from .modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb


def index_first_axis(x, indices):
    other_shape = x.shape[1:]
    second_dim = other_shape.numel()
    return torch.gather(
        rearrange(x, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim),
    ).reshape(-1, *other_shape)


def index_put_first_axis(x, indices, first_axis_dim):
    y = torch.zeros(first_axis_dim, *x.shape[1:], device=x.device, dtype=x.dtype)
    # TODO [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
    y[indices] = x
    # y.scatter_(0, repeat(indices, 'z -> z d', d=x.shape[1]), x)
    return y


@tensor_cache
def get_unpad_data(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    lens = prepare_lens_from_mask(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = lens.max().item()
    cu_seqlens = prepare_cu_seqlens_from_mask(attention_mask)
    return indices, cu_seqlens, max_seqlen_in_batch


def unpad_input(
    q: torch.Tensor,
    states: tuple[torch.Tensor],
    attention_mask: torch.Tensor,
    q_len: int,
    keepdim: bool = False,
):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = get_unpad_data(attention_mask)
    batch_size, seq_len, *_ = states[0].shape

    state = tuple(
        index_first_axis(rearrange(s, "b s ... -> (b s) ..."), indices_k)
        for s in states
    )

    if q_len == seq_len:
        q = index_first_axis(rearrange(q, "b s ... -> (b s) ..."), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif q_len == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
        indices_q = cu_seqlens_q[:-1]
        q = q.squeeze(1)
    else:
        raise NotImplementedError("We only support either q_len == k_len (prefilling) or q_len == 1 (decoding)")

    if keepdim:
        q = q.unsqueeze(0)
        state = tuple(s.unsqueeze(0) for s in state)

    return (
        q,
        state,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.LongTensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    output = index_put_first_axis(hidden_states, indices, batch_size * seq_len)
    return rearrange(output, "(b s) ... -> b s ...", b=batch_size)


class KimiDeltaAttention(nn.Module):
    def __init__(self, config: HypeNetConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.mode = "chunk"

        self.hidden_size = config.hidden_size
        self.head_dim = config.kda_head_dim
        self.num_heads = config.kda_num_heads
        self.head_k_dim = self.head_dim
        self.num_k_heads = self.num_heads
        self.use_conv = config.kda_use_conv
        self.use_qk_norm = config.kda_use_qk_norm
        self.use_rope = config.kda_use_rope

        self.layer_idx = layer_idx

        assert self.mode in [
            'chunk', 'fused_recurrent'], f"Not suppoerted mode `{self.mode}`."

        projection_k_size = self.head_k_dim * self.num_k_heads
        projection_size = self.head_dim * self.num_heads

        self.q_proj = nn.Linear(
            self.hidden_size, projection_k_size, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, projection_k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        if self.use_qk_norm:
            self.q_norm = Qwen3RMSNorm(
                self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen3RMSNorm(
                self.head_dim, eps=config.rms_norm_eps)

        if self.use_conv:
            self.conv_size = self.config.kda_conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=projection_k_size,
                kernel_size=self.conv_size,
                activation='silu',
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=projection_k_size,
                kernel_size=self.conv_size,
                activation='silu',
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=projection_size,
                kernel_size=self.conv_size,
                activation='silu',
            )

        self.A_log = torch.nn.Parameter(torch.log(torch.empty(
            self.num_heads, dtype=torch.float32).uniform_(1, 16)).view(1, 1, -1, 1))

        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.dt_bias = nn.Parameter(
            torch.empty(projection_size, dtype=torch.float32))

        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=config.rms_norm_eps, activation='sigmoid')
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: HybridCache | None = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, HybridCache | None]:
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                attention_mask = kwargs.get("padding_mask")

            if attention_mask is not None and attention_mask.dim() != 2:
                raise ValueError(
                    "attention_mask must be a 0-1 matrix of shape [batch_size, seq_len] "
                    "(0 = padding). 3D masks are not supported here.",
                )
        use_cache = past_key_values is not None
        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if q_len <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        cu_seqlens = kwargs.get('cu_seqlens')
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        conv_state_q, conv_state_k, conv_state_v = None, None, None

        if self.use_conv:
            # Get convolution states from cache
            if past_key_values is not None and len(past_key_values) > self.layer_idx:
                conv_state_q, conv_state_k, conv_state_v = past_key_values[self.layer_idx]['conv_state']

            # Compute short conv
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        g = self.f_b_proj(self.f_a_proj(hidden_states))
        g = fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)
        beta = self.b_proj(hidden_states).float().sigmoid()

        q, k = map(lambda x: rearrange(
            x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope:
            assert (
                position_embeddings is not None
            ), "position_embeddings is required when use_rope is True"
            cos, sin = position_embeddings
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            q, k = q.transpose(1, 2), k.transpose(1, 2)

        # Get recurrent state from cache
        recurrent_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            recurrent_state = past_key_values[self.layer_idx]['recurrent_state']
        if mode == 'chunk':
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx,
            )

        g = self.g_b_proj(self.g_a_proj(hidden_states))
        g = rearrange(g, '... (h d) -> ... h d', d=self.head_dim)
        o = self.o_norm(o, g)

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, None



def build_kda_with_attn(
    attn_layer: nn.Module,
    config: HypeNetConfig,
    layer_idx: int,
) -> nn.Module:

    layer = KimiDeltaAttention(
        config=config,
        layer_idx=layer_idx,
    )

    # print('============ Lighting attention layer ============')
    # print(f"Layer idx: {layer_idx}")
    # print(layer)
    # print('==================================================')

    if config.rand_init:
        return layer

    q_proj = attn_layer.q_proj
    k_proj = attn_layer.k_proj
    v_proj = attn_layer.v_proj
    o_proj = attn_layer.o_proj

    # (nh * head_dim, hidden_size)
    wq = q_proj.weight.data.clone()  # type: ignore
    wk = k_proj.weight.data.clone()  # type: ignore
    wv = v_proj.weight.data.clone()  # type: ignore
    wo = o_proj.weight.data.clone()  # type: ignore

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

    # print(layer)
    # print(wq.shape)
    # print(wk.shape)
    # print(wv.shape)
    # print(wo.shape)
    # print(layer.q_proj.weight.shape)
    # print(layer.k_proj.weight.shape)
    # print(layer.v_proj.weight.shape)
    # print(layer.o_proj.weight.shape)
    # exit()

    layer.q_proj.weight.data.copy_(wq)
    layer.k_proj.weight.data.copy_(wk)
    layer.v_proj.weight.data.copy_(wv)
    layer.o_proj.weight.data.copy_(wo)

    if hasattr(attn_layer, 'k_norm'):
        k_norm_weights = attn_layer.k_norm.weight.data.clone()
        layer.k_norm.weight.data.copy_(k_norm_weights)

    if hasattr(layer, 'q_norm'):
        q_norm_weights = attn_layer.q_norm.weight.data.clone()
        layer.q_norm.weight.data.copy_(q_norm_weights)

    return layer
