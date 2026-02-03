import os
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from modeling.hypenet.modeling_hypenet import HypeNetForCausalLM
from modeling.hypenet.configuration_hypenet import HypeNetConfig


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def run_inference(model, tokenizer, name, device):
    """Run a deterministic generation for sanity checking."""
    model.eval()
    model.to(device)

    prompt = "My name is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n=== {name} ===")
    print(text)
    print("=" * 60)
    return text


def load_rnn_checkpoint(ckpt_path):
    """
    Load and normalize an RNN checkpoint.

    Handles:
    - student_layer.*        -> *
    - model.model.*          -> model.*
    - model.lm_head          -> lm_head
    - optional lm_head tying
    """
    raw_state = load_file(ckpt_path)
    state = {}

    for k, v in raw_state.items():
        # drop teacher weights if present
        if "teacher_layer." in k:
            continue

        k = (
            k.replace("student_layer.", "")
             .replace("model.model.", "model.")
             .replace("model.lm_head", "lm_head")
        )
        state[k] = v

    # tie lm_head if missing
    if "lm_head.weight" not in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]

    return state


# ---------------------------------------------------------------------
# Hybrid construction
# ---------------------------------------------------------------------

def create_hybrid_model(
    attention_model,
    rnn_only_model,
    base_config_path,
    rnn_type,
    layer_indices,
):
    """
    Create a hybrid model by replacing selected layers with RNN mixers.

    NOTE:
    - All models use the same HypeNetForCausalLM class
    - Behavior is controlled purely by config.mixer_types
    """
    config = HypeNetConfig.from_json_file(base_config_path)

    # default: attention everywhere
    config.mixer_types = ["attn"] * config.num_hidden_layers

    # replace selected layers with RNN mixer
    for idx in layer_indices:
        config.mixer_types[idx] = rnn_type

    hybrid_model = HypeNetForCausalLM(config)
    hybrid_state = hybrid_model.state_dict()

    # 1) Non-layer parameters from attention model
    for k, v in attention_model.state_dict().items():
        if "model.layers." not in k:
            hybrid_state[k] = v

    # 2) Attention layers (layers NOT replaced)
    for k, v in attention_model.state_dict().items():
        if "model.layers." in k:
            layer_id = int(k.split(".")[2])
            if layer_id not in layer_indices:
                hybrid_state[k] = v

    # 3) RNN layers (layers replaced)
    for k, v in rnn_only_model.state_dict().items():
        if "model.layers." in k:
            layer_id = int(k.split(".")[2])
            if layer_id in layer_indices:
                hybrid_state[k] = v

    hybrid_model.load_state_dict(hybrid_state, strict=True)
    return hybrid_model


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(
    orig_model,
    rnn_base_path,
    ckpt_num,
    output_root,
    layer_indices,
    rnn_type,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # paths
    rnn_ckpt_path = f"{rnn_base_path}/ckpt_{ckpt_num}/model.safetensors"
    base_config_path = f"{rnn_base_path}/orig_config.json"

    print(f"[INFO] Attention base model : {orig_model}")
    print(f"[INFO] RNN checkpoint       : {rnn_ckpt_path}")
    print(f"[INFO] Replace layers       : {layer_indices}")
    print(f"[INFO] RNN mixer type        : {rnn_type}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(orig_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Attention-only model
    # ------------------------------------------------------------------
    attention_model = HypeNetForCausalLM.from_pretrained(
        orig_model,
        torch_dtype=torch.bfloat16,
    )
    run_inference(attention_model, tokenizer, "Attention model", device)

    # ------------------------------------------------------------------
    # RNN-only model
    # ------------------------------------------------------------------
    rnn_config = HypeNetConfig.from_json_file(rnn_base_path)
    rnn_config.mixer_types = [rnn_type] * rnn_config.num_hidden_layers

    rnn_only_model = HypeNetForCausalLM(rnn_config)

    rnn_state = load_rnn_checkpoint(rnn_ckpt_path)
    rnn_only_model.load_state_dict(rnn_state, strict=True)

    run_inference(rnn_only_model, tokenizer, "RNN-only model", device)

    # ------------------------------------------------------------------
    # Hybrid model
    # ------------------------------------------------------------------
    hybrid_model = create_hybrid_model(
        attention_model=attention_model,
        rnn_only_model=rnn_only_model,
        base_config_path=base_config_path,
        rnn_type=rnn_type,
        layer_indices=layer_indices,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(output_root, exist_ok=True)
    tokenizer.save_pretrained(output_root)
    hybrid_model.save_pretrained(output_root, safe_serialization=True)

    print(f"✅ Hybrid model saved to: {output_root}")

    run_inference(hybrid_model, tokenizer, "Hybrid model", device)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export a hybrid HypeNet model with selected RNN layers."
    )
    parser.add_argument("--orig_model", type=str, required=True)
    parser.add_argument("--rnn_base_path", type=str, required=True)
    parser.add_argument("--ckpt_num", type=int, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument(
        "--layer_indices",
        type=str,
        default="4",
        help="Comma-separated layer indices, e.g. 0,4,8",
    )
    parser.add_argument(
        "--rnn_type",
        type=str,
        default="lightning-attn",
        help="Mixer type used for RNN layers",
    )

    args = parser.parse_args()
    layer_indices = [int(x) for x in args.layer_indices.split(",") if x]

    main(
        orig_model=args.orig_model,
        rnn_base_path=args.rnn_base_path,
        ckpt_num=args.ckpt_num,
        output_root=args.output_root,
        layer_indices=layer_indices,
        rnn_type=args.rnn_type,
    )
