from pathlib import Path
import json

import torch
from torch import nn, Tensor
from safetensors.torch import load_file
from accelerate import Accelerator

from .modeling_hypenet import HypeNetForCausalLM
from .configuration_hypenet import HypeNetConfig
from arguments import Args


def load_state_dict(ckpt_dir: Path) -> dict[str, Tensor]:  # type: ignore
    # Try to load from a safetensors checkpoint.
    safetensors_path = ckpt_dir / 'model.safetensors'
    if safetensors_path.exists():
        return load_file(safetensors_path)
    
    # Try to load from a FSDP checkpoint.
    fsdp_path = ckpt_dir / 'pytorch_model_fsdp.bin'
    if fsdp_path.exists():
        state_dict = torch.load(fsdp_path, map_location="cpu")
        return state_dict

    # Try to load from a DeepSpeed checkpoint.
    ds_ckpt_dir = ckpt_dir / 'output_dir'
    ds_index_path = ds_ckpt_dir / 'pytorch_model.bin.index.json'
    if ds_index_path.exists():
        with open(ds_index_path, "r") as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        unique_files = set(weight_map.values())

        state_dict = {}

        for shard_file in sorted(unique_files):  # Sorting is optional but helpful
            shard_path = ds_ckpt_dir / shard_file
            shard = torch.load(shard_path, map_location="cpu")
            state_dict.update(shard)

        return state_dict

    raise ValueError(f"No checkpoint found in {ckpt_dir}")


def build_hybrid_from_ckpt(args: Args, accelerator: Accelerator, checkpoint_path: str) -> HypeNetForCausalLM:
    ckpt_path = Path(checkpoint_path)
    student_config_path = ckpt_path.parent / 'student_config.json'
    accelerator.print(f'Loading student config from {student_config_path}...')

    config = HypeNetConfig.from_json_file(student_config_path)
    accelerator.print("======== Config ==========")
    accelerator.print(config)
    accelerator.print("==========================")
    accelerator.print(f"Instantiating HybridForCausalLM...")
    # Must be BF16 for FSDP.
    model = HypeNetForCausalLM(config=config).to(torch.bfloat16)  # type: ignore

    accelerator.print(f"Loading state dict from {ckpt_path}...")
    state_dict = load_state_dict(ckpt_path)
    new_state_dict = {}
    for key, val in state_dict.items():
        if 'teacher_model' in key:
            # Just remove teacher weights.
            # TODO: I think we shouldn't have stored teacher weights in the first place.
            continue
        key = key.replace('student_model.', '')
        key = key.replace('.embeddings.', '.embed_tokens.')
        new_state_dict[key] = val

    accelerator.print("Tying word embeddings...")
    new_state_dict['lm_head.weight'] = new_state_dict['model.embed_tokens.weight']
    accelerator.print(f"Loading {len(new_state_dict)} parameters into model...")
    model.load_state_dict(new_state_dict)
    accelerator.print("Done!")

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    return model


def get_model(args: Args, config_path: str | None = None) -> nn.Module:
    if args.init_from not in ['scratch', 'none', '']:
        model = HypeNetForCausalLM.from_pretrained(args.init_from)
    elif config_path is not None:
        config = HypeNetConfig.from_json_file(config_path)
        model = HypeNetForCausalLM(config=config)
    else:
        raise ValueError("Either `args.init_from` or `config_path` must be provided.")

    model = model.to(torch.bfloat16)  # type: ignore

    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    return model
