from typing import Optional, Callable, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .modeling_qwen3 import Qwen3Attention, Qwen3ForCausalLM
from transformers.generation.utils import GenerationMixin


class LayerAligner(nn.Module):
    def __init__(self, teacher_layer: nn.Module, student_layer: nn.Module):
        super().__init__()
        self.teacher_layer = teacher_layer
        self.student_layer = student_layer

        # Only train the student layer
        self.teacher_layer.requires_grad_(False)
        self.student_layer.requires_grad_(True)

        self.teacher_hidden_states = None
        self.student_hidden_states = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        with torch.no_grad():
            teacher_outputs = self.teacher_layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                *args,
                **kwargs,
            )

        # detach hidden states to avoid gradient flow
        student_outputs = self.student_layer(
            hidden_states=hidden_states.detach(),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )

        # Cache student's output hidden states for alignment
        self.teacher_hidden_states = teacher_outputs[0].detach()
        self.student_hidden_states = student_outputs[0]

        outputs = (teacher_outputs[0].detach(), teacher_outputs[1], teacher_outputs[2])
        return outputs


class HiddenStateAligner(nn.Module, GenerationMixin):
    """
    This module is for performing "Hidden State Alignment",
    which swaps a block in a model, and aligns the outputs with
    L2 norm/MSE.

    This is a wrapper around a ...CausalLM model from
    HF transformers, which is passed in with the `model` argument.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_init_fn: Callable,
        loss_fn: str = 'l2norm',  # l2norm, mse
        convert_layer_idxs: Optional[list[int]] = None,
    ):
        super().__init__()
        self.model = model
        self.layer_init_fn = layer_init_fn
        self.loss_fn = loss_fn

        n_layers = len(self.model.model.layers)
        if convert_layer_idxs is None:
            convert_layer_idxs = list(range(n_layers))
        self.convert_layer_idxs = convert_layer_idxs
        self.add_student_layers(convert_layer_idxs)

        # Turn off gradients, and turn on gradients for the student layers
        self.requires_grad_(False)
        for name, param in self.model.named_parameters():
            if "student_layer" in name:
                param.requires_grad_(True)

    def add_student_layers(self, layer_idxs: list[int]):
        for layer_idx in layer_idxs:
            teacher_layer: Qwen3Attention = self.model.model.layers[layer_idx].self_attn  # type: ignore
            student_layer = self.layer_init_fn(teacher_layer, layer_idx=layer_idx)
            aligner = LayerAligner(teacher_layer=teacher_layer, student_layer=student_layer)
            self.model.model.layers[layer_idx].self_attn = aligner

    def get_loss(
        self,
        teacher_hidden_states: torch.Tensor,
        student_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.loss_fn == 'l2norm':
            dim = teacher_hidden_states.shape[-1]
            scale = dim ** -0.5
            alignment_loss = torch.linalg.vector_norm(teacher_hidden_states - student_hidden_states, dim=-1)
            alignment_loss = alignment_loss.float().mean() * scale
            return alignment_loss
        elif self.loss_fn == 'mse':
            return F.mse_loss(teacher_hidden_states, student_hidden_states)
        else:
            raise ValueError(f'Invalid loss function: {self.loss_fn}')

    def forward(
        self,
        *args,
        **kwargs,
    ):
        outputs = self.model(*args, **kwargs)

        # Get alignment loss
        aligners: List[LayerAligner] = [self.model.model.layers[layer_idx].self_attn for layer_idx in self.convert_layer_idxs]  # type: ignore
        # (L, B, T, D)
        teacher_hidden_states = torch.stack([aligner.teacher_hidden_states for aligner in aligners], dim=0)
        student_hidden_states = torch.stack([aligner.student_hidden_states for aligner in aligners], dim=0)
        loss = self.get_loss(teacher_hidden_states, student_hidden_states)

        # for i in range(len(teacher_hidden_states)):
        #     mean_t = teacher_hidden_states[i].mean().item()
        #     std_t = teacher_hidden_states[i].std().item()
        #     min_t = teacher_hidden_states[i].min().item()
        #     max_t = teacher_hidden_states[i].max().item()
        #     mean_s = student_hidden_states[i].mean().item()
        #     std_s = student_hidden_states[i].std().item()
        #     min_s = student_hidden_states[i].min().item()
        #     max_s = student_hidden_states[i].max().item()
        #     print(f'teacher {i} mean: {mean_t:.4e}\tstd: {std_t:.4e}\tmin: {min_t:.4e}\tmax: {max_t:.4e}')
        #     print(f'student {i} mean: {mean_s:.4e}\tstd: {std_s:.4e}\tmin: {min_s:.4e}\tmax: {max_s:.4e}')
        # print('teacher has nan:', torch.isnan(teacher_hidden_states).any())
        # print('student has nan:', torch.isnan(student_hidden_states).any())
        # print('alignment loss:', alignment_loss.item())
        # print('max diff:', (teacher_hidden_states - student_hidden_states).abs().max())
        # print('teacher max:', (teacher_hidden_states).abs().max())
        # print('student max:', (student_hidden_states).abs().max())

        return loss


if __name__ == "__main__":
    from transformers import Qwen3ForCausalLM

    model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-4B")
    gdn = init_gdn_with_attn(model.model.layers[0].self_attn, 0)
