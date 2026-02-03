############################
# User-configurable paths
############################
ORIG_MODEL=${ORIG_MODEL:-/path/to/original/model}
RNN_BASE_PATH=${RNN_BASE_PATH:-/path/to/rnn/checkpoints}
CKPT_NUM=${CKPT_NUM:-20000}

MODEL_TYPE=${MODEL_TYPE:-hypenet-2b-lightning}
RNN_TYPE=${RNN_TYPE:-lightning-attn}

############################
# Layer sweep config
############################
START_LAYER=${START_LAYER:-0}
END_LAYER=${END_LAYER:-27}

############################
# Results directory
############################
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE_DIR=${RESULTS_BASE_DIR:-results/layer_sweep_${TIMESTAMP}}

mkdir -p "${RESULTS_BASE_DIR}"

echo "=============================================="
echo "Layer sweep started at $(date)"
echo "Model type     : ${MODEL_TYPE}"
echo "RNN type       : ${RNN_TYPE}"
echo "Layers         : baseline + ${START_LAYER} → ${END_LAYER}"
echo "Results dir    : ${RESULTS_BASE_DIR}"
echo "=============================================="

########################################
# Baseline (original model) evaluation
########################################
run_baseline_evaluation () {
    local layer_dir="${RESULTS_BASE_DIR}/original"

    mkdir -p "${layer_dir}"

    echo "[Baseline] Running commonsense eval..."

    if ! accelerate launch -m lm_eval \
        --model hf \
        --model_args "pretrained=${ORIG_MODEL},dtype=bfloat16" \
        --tasks hellaswag,arc_easy,arc_challenge \
        --num_fewshot 0 \
        --batch_size 16 \
        > "${layer_dir}/commonsense_eval.log" 2>&1; then

        echo "[Baseline] ❌ Commonsense eval failed" | tee "${layer_dir}/error.log"
    fi

    echo "[Baseline] Running retrieval eval..."

    if ! accelerate launch -m lm_eval \
        --model hf \
        --model_args "pretrained=${ORIG_MODEL},dtype=bfloat16" \
        --tasks ruler_qa_squad,fda,swde \
        --num_fewshot 0 \
        > "${layer_dir}/retrieval_eval.log" 2>&1; then

        echo "[Baseline] ❌ Retrieval eval failed" >> "${layer_dir}/error.log"
    fi

    echo "[Baseline] ✅ Done"
}

########################################
# Layer conversion + evaluation
########################################
run_layer_evaluation () {
    local layer=$1

    local model_name="${MODEL_TYPE}_${RNN_TYPE}/layer${layer}"
    local output_root="models/${model_name}"
    local layer_dir="${RESULTS_BASE_DIR}/${model_name}"

    mkdir -p "${layer_dir}"

    echo "[Layer ${layer}] Converting model..."

    if ! python convert.py \
        --orig_model "${ORIG_MODEL}" \
        --rnn_base_path "${RNN_BASE_PATH}" \
        --ckpt_num "${CKPT_NUM}" \
        --output_root "${output_root}" \
        --layer_indices "${layer}" \
        --rnn_type "${RNN_TYPE}" \
        > "${layer_dir}/conversion.log" 2>&1; then

        echo "[Layer ${layer}] ❌ Conversion failed" | tee "${layer_dir}/error.log"
        return 1
    fi

    echo "[Layer ${layer}] Running commonsense eval..."

    if ! accelerate launch -m lm_eval \
        --model hf \
        --model_args "pretrained=${output_root},dtype=bfloat16" \
        --tasks hellaswag,arc_easy,arc_challenge \
        --num_fewshot 0 \
        --batch_size 16 \
        > "${layer_dir}/commonsense_eval.log" 2>&1; then

        echo "[Layer ${layer}] ❌ Commonsense eval failed" >> "${layer_dir}/error.log"
    fi

    echo "[Layer ${layer}] Running retrieval eval..."

    if ! accelerate launch -m lm_eval \
        --model hf \
        --model_args "pretrained=${output_root},dtype=bfloat16" \
        --tasks ruler_qa_squad,fda,swde \
        --num_fewshot 0 \
        > "${layer_dir}/retrieval_eval.log" 2>&1; then

        echo "[Layer ${layer}] ❌ Retrieval eval failed" >> "${layer_dir}/error.log"
    fi

    echo "[Layer ${layer}] ✅ Done"
}

########################################
# Run baseline first
########################################
echo "=============================================="
echo "Running baseline (original model)"
echo "=============================================="

run_baseline_evaluation

########################################
# Main layer sweep
########################################
for (( layer=START_LAYER; layer<=END_LAYER; layer++ )); do
    echo "----------------------------------------------"
    echo "Starting layer ${layer}"
    echo "----------------------------------------------"

    if ! run_layer_evaluation "${layer}"; then
        echo "[Layer ${layer}] ⚠️  Failed, continuing..."
    fi
done

########################################
# Summary
########################################
SUMMARY_FILE="${RESULTS_BASE_DIR}/summary.txt"

{
    echo "Layer Sweep Summary"
    echo "Timestamp   : ${TIMESTAMP}"
    echo "Model type  : ${MODEL_TYPE}"
    echo "RNN type    : ${RNN_TYPE}"
    echo "Layers      : baseline + ${START_LAYER} → ${END_LAYER}"
    echo ""

    if [[ -f "${RESULTS_BASE_DIR}/original/error.log" ]]; then
        echo "Baseline (original): FAILED"
    else
        echo "Baseline (original): OK"
    fi

    for (( layer=START_LAYER; layer<=END_LAYER; layer++ )); do
        model_name="${MODEL_TYPE}_${RNN_TYPE}/layer${layer}"
        layer_dir="${RESULTS_BASE_DIR}/${model_name}"

        if [[ -f "${layer_dir}/error.log" ]]; then
            echo "Layer ${layer}: FAILED"
        else
            echo "Layer ${layer}: OK"
        fi
    done
} > "${SUMMARY_FILE}"

echo "=============================================="
echo "Layer sweep completed at $(date)"
echo "Summary: ${SUMMARY_FILE}"
echo "=============================================="
