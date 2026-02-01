"""
Used in Stage 2 of HALO.

This code implements a Module for performing distillation between
two `CausalLM` models using KL divergence.
"""

from pathlib import Path
import copy

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from safetensors.torch import load_file
from accelerate import Accelerator

from .modeling_hypenet import HypeNetForCausalLM
from .configuration_hypenet import HypeNetConfig
from arguments import Args


class DistillationModel(nn.Module):
    '''
    A module for distillation between two `CausalLM` models.

    Args:
        teacher_model: The teacher model.
        student_model: The student model.
        loss_fn: The loss function to use for distillation.
    '''
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        loss_fn: str = 'kl_div',
    ):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model

        self.loss_fn = loss_fn
        assert self.loss_fn in ['kl_div', 'mse']

        # Only train the student model.
        self.teacher_model.requires_grad_(False)
        self.student_model.requires_grad_(True)

    def compute_loss(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        # (bsz * seq_len, vocab_size)
        # print(student_logits.shape, teacher_logits.shape)
        flat_student_logits = student_logits.view(-1, student_logits.shape[-1])
        flat_teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])

        if self.loss_fn == 'kl_div':
            log_prob_student = F.log_softmax(flat_student_logits, dim=-1)
            log_prob_teacher = F.log_softmax(flat_teacher_logits, dim=-1)

            # Compute KL divergence in chunk-wise manner, because
            # somehow `kl_div` is very memory-intensive.
            distill_loss = torch.tensor(
                0.0, device=flat_student_logits.device, dtype=flat_student_logits.dtype
            )
            chunk_len = 512
            n_chunks = (flat_student_logits.shape[0] + chunk_len - 1) // chunk_len
            for i in range(0, n_chunks, chunk_len):
                distill_loss += F.kl_div(
                    log_prob_student[i : i + chunk_len],
                    log_prob_teacher[i : i + chunk_len],
                    log_target=True,
                    reduction="sum",
                )
            loss = distill_loss / flat_student_logits.shape[0]  # Maybe add CE loss?
        elif self.loss_fn == 'mse':
            loss = F.mse_loss(flat_student_logits, flat_teacher_logits)
        else:
            raise ValueError(f"Invalid loss function: {self.loss_fn}")
        return loss

    def forward(self, *args, **kwargs) -> Tensor:
        with torch.no_grad():
            teacher_outputs = self.teacher_model(*args, **kwargs, return_logits=True)
        student_outputs = self.student_model(*args, **kwargs, return_logits=True)

        # Compute KL divergence between teacher and student outputs
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits.detach()

        loss = self.compute_loss(student_logits, teacher_logits)
        return loss


def build_hybrid_from_ckpt(
    args: Args,
    accelerator: Accelerator,
    model_name: str,
    checkpoint_path: str,
) -> HypeNetForCausalLM:
    ckpt_path = Path(checkpoint_path)
    teacher_config_path = ckpt_path.parent / "orig_config.json"
    accelerator.print(f"Loading teacher config from {teacher_config_path}")
    orig_config: HypeNetConfig = HypeNetConfig.from_json_file(teacher_config_path)  # type: ignore
    accelerator.print(f"==== teacher config ====")
    accelerator.print(orig_config)
    accelerator.print(f"========================")

    # Create student model based on teacher config.
    accelerator.print(f"Creating student model based on teacher config...")
    assert args.model_config is not None, "model_config is required"
    student_config = HypeNetConfig.from_json_file(args.model_config)

    # Create student model.
    student_model: nn.Module = HypeNetForCausalLM(student_config).to(dtype=torch.bfloat16)

    accelerator.print(f"==== student config ====")
    accelerator.print(student_config)
    accelerator.print(f"========================")

    # Load and convert state dict
    accelerator.print(f"Loading parameters from {ckpt_path}...")
    state_dict = load_file(ckpt_path / "model.safetensors")

    accelerator.print(f"Loading parameters into student model...")
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        if ".layers." in key:
            layer_idx = int(key.split('.layers.')[1].split('.')[0])
            mixer_type = student_config.mixer_types[layer_idx]
            if mixer_type == 'attn':
                # For attention layers, keep the teacher layer (which is the
                # original attention layer)
                if 'student_layer' in key:
                    continue
                # Remove teacher_layer prefix.
                key = key.replace("teacher_layer.", "")
            else:
                # For RNN layers, keep the student layer (which is the
                # distilled attention layer)
                if 'teacher_layer' in key:
                    continue
                # Remove student_layer prefix.
                key = key.replace("student_layer.", "")
        new_state_dict[key] = value
    new_state_dict["lm_head.weight"] = new_state_dict["model.embed_tokens.weight"]
    missing_keys, unexpected_keys = student_model.load_state_dict(new_state_dict, strict=False)
    assert len(unexpected_keys) == 0
    for missing_key in missing_keys:
        # Allow adding new parameters to teacher layers
        if student_config.attn_use_output_gate:
            # This adds a `gate_proj` to the teacher layer.
            if '.self_attn.o_gate.' in missing_key:
                continue

        raise ValueError(f"Missing key: {missing_key}")
    n_params = sum(p.numel() for p in student_model.parameters())
    accelerator.print(f"Student model loaded with {n_params:,} parameters.")

    if args.grad_ckpt:
        student_model.gradient_checkpointing_enable()

    return student_model


def build_gdn_from_ckpt(
    args: Args,
    accelerator: Accelerator,
    model_name: str,
    checkpoint_path: str,
) -> HypeNetForCausalLM:
    """
    Create a `HybridForCausalLM` model from the checkpoint
    from hidden state alignment (see `./hidden_state_alignment.py`).

    This will remove the teacher layers (usually Qwen3Attention layers)
    from the checkpoint, and instantiate a standalone hybrid model with
    the student layers (it may be a fully RNN model).
    """
    ckpt_path = Path(checkpoint_path)
    teacher_config_path = ckpt_path.parent / "orig_config.json"
    accelerator.print(f"Loading teacher config from {teacher_config_path}")
    orig_config: HypeNetConfig = HypeNetConfig.from_json_file(teacher_config_path)  # type: ignore

    # Create student model based on teacher config.
    accelerator.print(f"Creating student model based on teacher config...")
    if args.model_config is None:
        # By default, convert all layers to GDN.
        new_config = copy.deepcopy(orig_config)
        new_config.mixer_types = ['gdn'] * new_config.num_hidden_layers
    else:
        # If model_config is provided, use it to create the student model.
        accelerator.print(f"Loading student config from {args.model_config}")
        new_config = HypeNetConfig.from_json_file(args.model_config)

    student_model = HypeNetForCausalLM(new_config).to(dtype=torch.bfloat16)

    accelerator.print(f"==== student config ====")
    accelerator.print(new_config)
    accelerator.print(f"========================")
    student_config_path = ckpt_path.parent / "student_config.json"
    accelerator.print(f"Saving student config to {student_config_path}")
    new_config.to_json_file(student_config_path)
    n_params = sum(p.numel() for p in student_model.parameters())
    accelerator.print(f"Student model has {n_params:,} parameters.")

    # Load parameters into student model.
    accelerator.print(f"Loading parameters from {ckpt_path}...")
    state_dict = load_file(ckpt_path / "model.safetensors")
    # print(list(state_dict.keys()))

    accelerator.print(f"Loading parameters into student model...")
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        if '.layers.' in key:
            layer_idx = int(key.split('.layers.')[1].split('.')[0])
            mixer_type = new_config.mixer_types[layer_idx]
            if mixer_type == 'gdn':
                if 'teacher_layer' in key:
                    # Remove "attention layers".
                    continue
                key = key.replace("student_layer.", "")  # Remove student_layer prefix.
                key = key.replace(".input_layernorm.", ".mixer_norm.")
                key = key.replace(".post_attention_layernorm.", ".mlp_norm.")
                key = key.replace(".self_attn.", ".mixer.")
                new_state_dict[key] = value
            elif mixer_type == 'attn':
                if 'student_layer' in key:
                    # Remove "GDN layers".
                    continue
                key = key.replace("teacher_layer.", "")  # Remove teacher_layer prefix.
                key = key.replace(".input_layernorm.", ".mixer_norm.")
                key = key.replace(".post_attention_layernorm.", ".mlp_norm.")
                key = key.replace(".self_attn.", ".mixer.")
                new_state_dict[key] = value
        else:
            new_state_dict[key] = value

    new_state_dict["lm_head.weight"] = new_state_dict["model.embed_tokens.weight"]

    student_model.load_state_dict(new_state_dict, strict=True)
    n_params = sum(p.numel() for p in student_model.parameters())
    accelerator.print(f"Student model loaded with {n_params:,} parameters.")

    if args.grad_ckpt:
        student_model.gradient_checkpointing_enable()

    return student_model
