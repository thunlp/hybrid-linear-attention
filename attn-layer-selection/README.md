# Attention Layer Selection in HALO

This directory contains the implementation for **attention layer selection** in **HALO (Hybrid Attention via Layer Optimization)**. In this stage, we evaluate the relative importance of each attention layer to determine which should be converted to **linear attention**.

Specifically, this directory handles:

* **Layer-wise Sweeping**: Assessing the contributions of individual layers.
* **Performance Profiling**: Measuring impacts on commonsense reasoning and long-context retrieval.
* **Importance Ranking**: Generating the scores that guide the final hybrid architecture selection.

---

## Environment

### Software

* **Python**: 3.12.12
* **PyTorch**: 2.9.1
* **Transformers**: 4.57.6
* **Accelerate**: 1.12.0
* **Flash-Linear-Attention**: 0.4.2
* **Flash-Attention**: 2.8.3
* **Triton**: 3.5.1
* **lm-evaluation-harness**: 0.4.9.1

### Hardware

* **OS**: Ubuntu 22.04
* **GPU**: 8 × NVIDIA A800-80GB

---

## Setup

1. **Install dependencies**:
```bash
pip install -r ../halo/requirements.txt

```


2. **Install lm-evaluation-harness**: Follow the official installation guide: [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
3. **Complete Stage 1 of HALO**: Attention layer selection requires the **RNN checkpoint produced in Stage 1**. Ensure your Stage 1 RNN checkpoint is ready before proceeding.

---

## Usage

### Step 1: Perform the Layer Sweep

This step evaluates model behavior when specific layers are replaced with linear attention. Edit `scripts/eval_hypenet.sh` with your local paths and configurations:

```bash
# Core Configuration
ORIG_MODEL=${ORIG_MODEL:-/path/to/original/model}
RNN_BASE_PATH=${RNN_BASE_PATH:-/path/to/rnn/checkpoints}
CKPT_NUM=${CKPT_NUM:-20000}
MODEL_TYPE=${MODEL_TYPE:-hypenet-2b}

# Sweep Range (e.g., for a 28-layer model)
START_LAYER=${START_LAYER:-0}
END_LAYER=${END_LAYER:-27}

# Output Directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE_DIR=${RESULTS_BASE_DIR:-results/layer_sweep_${TIMESTAMP}}

```

Execute the sweep:

```bash
bash scripts/eval_hypenet_layerwise.sh

```

This script:

* Performs a **layer-wise sweep** across all attention layers.
* Stores results in the following structure:
  * `{RESULTS_BASE_DIR}/{MODEL_TYPE}/layer{L}/` for each individual layer.
  * `{RESULTS_BASE_DIR}/original/` for the baseline (original) model.


---

### Step 2: Compute Importance Scores

After completing the sweep, analyze the data to identify the optimal layers for conversion:

```bash
python layer_analysis.py \
  --root {RESULTS_BASE_DIR} \
  --model {MODEL_TYPE} \
  --mode HALO

```

**Alternatively**, use the provided shortcut:

```bash
bash scripts/layer_analysis.sh

```

The output provides **sorted importance scores** for each layer:

* **Higher Scores**: High importance (retain as standard attention).
* **Lower Scores**: Candidates for linearization (convert to linear attention).

---

## Reproducing HypeNet Results

We provide pre-computed results for the HypeNet family (2B, 4B, and 8B) in the `results/` directory. You can skip the compute-intensive sweep and run **Step 2** directly on these folders to see how we derived our specific architectures.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{hybrid-linear-attention-done-right,
  title        = {Hybrid Linear Attention Done Right: Efficient Distillation and Effective Architectures for Extremely Long Contexts},
  author       = {Yingfa Chen and Zhen Leng Thai and Zihan Zhou and Zhu Zhang and Xingyu Shen and Shuo Wang and Chaojun Xiao and Xu Han and Zhiyuan Liu},
  year         = {2026},
  eprint       = {2601.22156},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2601.22156}
}

```