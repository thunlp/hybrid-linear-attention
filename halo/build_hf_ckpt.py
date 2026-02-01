import torch
from modeling.tfm2rnn import HypeNetForCausalLM, HybridConfig
from transformers import AutoTokenizer
from torch import Tensor
from pathlib import Path
from safetensors.torch import load_file
import json
from tap import Tap


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
    ds_index_path = ckpt_dir / 'output_dir/pytorch_model.bin.index.json'
    if ds_index_path.exists():
        with open(ds_index_path, "r") as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        unique_files = set(weight_map.values())

        state_dict = {}

        for shard_file in sorted(unique_files):  # Sorting is optional but helpful
            shard_path = ckpt_dir / 'output_dir' / shard_file
            shard = torch.load(shard_path, map_location="cpu")
            state_dict.update(shard)

        return state_dict

    raise ValueError(f"No checkpoint found in {ckpt_dir}")


class Args(Tap):
    only_state_dict: int = 0
    stage: int = 3
    tok_path = '/path/to/qwen3-1.7b'
    ckpt_path = "/home/test/test07/chenyingfa/tfm2rnn/tiny-pretrainer/results/hypenet/e7_stage3_hypenet-1.7b-gdn-attn-eq0_stage3_"
    config_path = "configs/model/hypenet/hypenet-1.7b-gdn-attn-eq0.json"
    out_path = "ckpts/hypenet-1.7b-gdn-attn-eq0_stage3"
    ckpt = "ckpt_500"


args = Args().parse_args()
ckpt_path = Path(args.ckpt_path) / args.ckpt
tok_path = Path(args.tok_path)
config_path = Path(args.config_path)
out_path = Path(args.out_path) / args.ckpt
out_path.mkdir(exist_ok=True, parents=True)
assert ckpt_path.exists(), f"Checkpoint {ckpt_path} does not exist"
assert config_path.exists(), f"Config {config_path} does not exist"
assert tok_path.exists(), f"Tokenizer {tok_path} does not exist"

print(f"Loading tokenizer from {tok_path}")
tokenizer = AutoTokenizer.from_pretrained(tok_path)
print(f"Loading config from {config_path}")
config = HybridConfig.from_json_file(config_path)

print(config)

print("Instantiating model")
model = HypeNetForCausalLM(config=config)

print(f"Loading state dict from {ckpt_path}")
state_dict = load_state_dict(ckpt_path)
print(list(state_dict.keys()))

if args.stage == 2:
    new_state_dict = {}
    for key, val in state_dict.items():
        if 'teacher_model' in key:
            # Just remove teacher weights.
            # TODO: I think we shouldn't have stored teacher weights in the first place.
            continue
        key = key.replace('student_model.', '')
        print(f"Inserting parameter {key}...")
        new_state_dict[key] = val.to(torch.bfloat16)
    state_dict = new_state_dict

if bool(args.only_state_dict):
    print(f"Saving state dict to {out_path}")
    torch.save(state_dict, out_path / 'state_dict.pt')
    exit()


if 'lm_head.weight' not in state_dict:
    assert config.tie_word_embeddings, "lm_head.weight is not in the state dict, so tie_word_embeddings must be True"
    # If there is no lm_head.weight, tie it to the word embeddings.
    print("Tying word embeddings...")
    state_dict['lm_head.weight'] = state_dict['model.embed_tokens.weight']

print("Loading parameters into model")
model.load_state_dict(state_dict)
model = model.to(torch.bfloat16)

print("Registering model")
model.config.register_for_auto_class()
model.model.register_for_auto_class("AutoModel")
model.register_for_auto_class("AutoModelForCausalLM")

print(f"Saving model to {out_path}")
model.save_pretrained(out_path)
print(f"Saving tokenizer to {out_path}")
tokenizer.save_pretrained(out_path)
