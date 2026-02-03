# Attention Layer Selection in HALO

This directory contains code for selecting attention layers in HALO (Hybrid Attention via Layer Optimization). This stage evaluates each attention layer's importance score and determines which layers should be converted to linear attention.

## Environment

**Software:**
- Python 3.12.12
- PyTorch 2.9.1
- Transformers 4.57.6
- Accelerate 1.12.0
- Flash-Linear-Attention 0.4.2
- Flash-Attention 2.8.3
- Triton 3.5.1
- lm-evaluation-harness 0.4.9.1

On a Ubuntu 22.04 system with 8 A800-80G GPUs.

## Setup

1. Install dependencies:
   ```bash
   pip install -r ../halo/requirements.txt
   ```

2. Install lm-evaluation-harness following the [official instructions](https://github.com/EleutherAI/lm-evaluation-harness)

3. Complete Stage 1 of HALO before running attention layer selection, as this procedure requires the RNN checkpoint from Stage 1

## Usage

### Step 1: Evaluate Commonsense and Retrieval Performance

Configure paths in `scripts/eval_hypenet.sh`:

```bash
# User-configurable paths
ORIG_MODEL=${ORIG_MODEL:-/path/to/original/model}
RNN_BASE_PATH=${RNN_BASE_PATH:-/path/to/rnn/checkpoints}
CKPT_NUM=${CKPT_NUM:-20000}

# Model configuration
MODEL_TYPE=${MODEL_TYPE:-hypenet-2b}
RNN_TYPE=${RNN_TYPE:-lightning-attn}

# Layer sweep configuration
# Set END_LAYER to num_layers - 1
START_LAYER=${START_LAYER:-0}
END_LAYER=${END_LAYER:-27}
```

Run the evaluation:

```bash
bash scripts/eval_hypenet.sh
```

This performs a layer-wise sweep and evaluates performance on commonsense reasoning and long-context retrieval tasks.

### Step 2: Compute Attention Layer Importance

Analyze layer importance scores:

```bash
python layer_analysis.py \
  --root /path/to/layer_sweep \
  --model hypenet-2b-lightning \
  --mode HALO
```

Or use the provided script:

```bash
bash scripts/layer_analysis.sh
```

The output displays sorted importance scores for each attention layer.

## Reproducing Results

Pre-computed layer sweep results for HypeNet-2B, HypeNet-4B, and HypeNet-8B are available in the `results/` directory. You can reproduce the importance scores by running Step 2 on these provided outputs.

## Citation

If you use this code, please cite:

```bibtex
@misc{hybrid-linear-attention-done-right,
  title={Hybrid Linear Attention Done Right: Efficient Distillation and Effective Architectures for Extremely Long Contexts},
  author={Yingfa Chen and Zhen Leng Thai and Zihan Zhou and Zhu Zhang and Xingyu Shen and Shuo Wang and Chaojun Xiao and Xu Han and Zhiyuan Liu},
  year={2026},
  eprint={2601.22156},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2601.22156}
}