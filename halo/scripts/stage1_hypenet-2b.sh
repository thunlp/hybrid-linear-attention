args="$@"
for arg in "$@"; do
    eval "$arg"
done

proj_name=${proj_name:-"hypenet"}
exp_group=${exp_group:-"e1"}
model_name=${model_name:-"LA"}
model_config=${model_config:-"hypenet-2b"}
train_config=${train_config:-"stage1"}
comment=${comment:-""}
orig_model=${orig_model:-"/path/to/qwen3-1.7b"}

run_name="${exp_group}_${model_config}_${train_config}_${comment}"

gpus_per_node=${gpus_per_node:-8}
n_machines=${n_machines:-1}
machine_rank=${machine_rank:-0}
master_port=${master_port:-16603}
n_gpus=$(( $n_machines * $gpus_per_node ))

cmd="accelerate launch"
cmd+=" --config_file=configs/accelerate/multigpu_config.yaml"
cmd+=" --main_process_port=$master_port"
cmd+=" --machine_rank=$machine_rank"
cmd+=" --num_processes=$n_gpus"
cmd+=" --num_machines=$n_machines"
cmd+=" stage1.py"
cmd+=" --proj_name=${proj_name}"
cmd+=" --model_name=${model_name}"
cmd+=" --run_name=${run_name}"
cmd+=" --train_config=configs/training/${train_config}.json"
cmd+=" --model_config=configs/model/hypenet/${model_config}.json"
cmd+=" --init_from=${orig_model}"
cmd+=" --tok_path=${orig_model}"
cmd+=" --stage=1"

echo "==== Final command ===="
echo "$cmd"
echo "======================="
nvidia-smi
echo "======================="
eval "$cmd"

echo "Stage 1 completed."
