# HALO

This directory contains code for implementing the stage 1, 2, and 3 of the HALO (Hybrid Attention via Layer Optimization) training procedure.

## Environment

The code is tested on:

- Python 3.12.12
- PyTorch 2.9.1
- Transformers 4.57.6
- Accelerate 1.12.0
- Flash-Linear-Attention 0.4.2
- Flash-Attention 2.8.3
- Triton 3.5.1

On a Ubuntu 22.04 system with 8 A800-80G GPUs.

## Training

The training consists of the following steps:

- Stage 1: Hidden state alignment
- Attention layer selection
- Stage 2: Knowledge distilation
- Stage 3: Finetuning

The following sections show an example of how to run this code for converting a Qwen3-1.7B checkpoint into HypeNet using the HALO training procedure.

> By default, the training log will be saved with Tensorboard and Swanlab. If you want to use Wandb (or other experiment trackers that HuggingFace Accelerate supports), you should set the `--report_to` argument in the following scripts.

**Step 0: Setup**

Please do the following to prepare for distillation.

1. Install the dependencies: `pip install -r requirements.txt`.
2. Download Qwen3-1.7B checkpoint from: <https://huggingface.co/Qwen/Qwen3-1.7B>
3. Download the FineWeb-Edu data from: <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>

**Step 1: Run Stage 1 of HALO**

This step starts by transferring the weights from Qwen3 to a hybrid model, then aligns the hidden states (outputs of token mixing layers) between Qwen3 and the hybrid model.

```bash
bash ./scripts/stage1_hypenet_2b.sh orig_model=/path/to/qwen3-1.7b data_path=/path/to/fineweb-edu-100bt
```

**Step 1.5 (optional): Attention Layer Selection**

> Unless you want to modify the attention layer selection method, you can directly use the attention layer indices we have pre-computed and defined in the model configs (i.e., the `mixer_type` field in `configs/model/hypenet/hypenet-2b.json` for example), and do not need to rerun the attenion layer selection experiments.

The selection method is implemented in a separate codebase. Please refer to `attn-layer-selection` to reproduce our experiments in this part.

**Step 2: Run stage 2 of HALO**

This step performs end-to-end distillation from Qwen3 to the hybrid model using KL divergence.

```bash
bash ./scripts/stage2_hypenet_2b.sh orig_model=/path/to/qwen3-1.7b data_path=/path/to/fineweb-edu-100bt
```

**Step 3: Run stage 3 of HALO**

This step finetunes the distilled model on using longer training contexts without distillation (no teacher model).

```bash
bash ./scripts/stage3_hypenet_2b.sh orig_model=/path/to/qwen3-1.7b data_path=/path/to/fineweb-edu-100bt
```

### Step 4: Extract the checkpoint into HuggingFace format

Use the following code to extract the checkpoint into HuggingFace format such that you can use `AutoModelForCausalLM.from_pretrained(...)` to load and test your model.
```shell
python build_hf_ckpt.py \
--ckpt_path=results/hypenet/e1_stage3_hypenet-2b_stage3_ \
--config_path=configs/model/hypenet/hypenet-2b.json \
--ckpt=ckpt_500 \
--stage=3 \
--out_path=ckpts/hypenet-2b
```

### Notes for Converting the 8B Model

> When converting the 1.7B and 4B Qwen3 models, you can ignore this section.

When converting the 8B model, stage 2 and stage 3 needs to use Zero3 to reduce the memory usage, therefore, you need to convert the DeepSpeed checkpoint to a PyTorch checkpoint after stage 2 and 3 with the following command in the checkpoint directory, which is located at `./results/hypenet/{run_name}/ckpt_{training_step}`.

```bash
python zero_to_fp32.py . output_dir
```

## Evaluation/Inference

> Currently, our model does not support batched inference, we are actively working on this, and will update the code once ready.

To test the final model, use a standard HF transformer code to load the final checkpoint.

```python
"""
A short code for performing inference with HF tranformers.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "ckpts/hypenet-2b"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()

prompts = ["My name is"]
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    padding_side="left",
).to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)

output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=False)
print(output_texts)
```

## How to Use Other RNN Mixers?

Currently, our HypeNet use Lightning Attention by default as the RNN mixer, as supported by the experimental results in the paper. You can also use other RNN mixers by:

- Changing the `model_name` command-line argument; and
- Changing the `mixer_type` field in the model config file.

For instance, if you want to use GDN (Gated DeltaNet) instead, you need to pass `--model_name=gdn` and make a model config where `mixer_type` contains `gdn` instead of `lightning-attn`.

## How to Cite?

The following is the BibTex for citing us.

```bibtex
@misc{hybrid-linear-attention-done-right,
      title={Hybrid Linear Attention Done Right: Efficient Distillation and Effective Architectures for Extremely Long Contexts}, 
      author={Yingfa Chen and Zhen Leng Thai and Zihan Zhou and Zhu Zhang and Xingyu Shen and Shuo Wang and Chaojun Xiao and Xu Han and Zhiyuan Liu},
      year={2026},
      eprint={2601.22156},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.22156}, 
}
```