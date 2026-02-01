args="$@"
for arg in "$@"; do
    eval "$arg"
done

gpus_per_node=${gpus_per_node:-8}
n_machines=${n_machines:-1}
machine_rank=${machine_rank:-0}
master_port=${master_port:-6603}
n_gpus=$(( $n_machines * $gpus_per_node ))

proj_name=${proj_name:-"hypenet"}
exp_group=${exp_group:-"e1"}
model_config=${model_config:-"hypenet-5b"}
train_config=${train_config:-"stage3"}
comment=${comment:-""}
tok_path=${tok_path:-"/path/to/qwen3-4b"}

stage2_ckpt=${stage2_ckpt:-"ckpt_20000"}
stage2_run_name=${stage2_run_name:-"${exp_group}_stage2_${model_config}_stage2_"}
path_stage2=${path_stage2:-"results/${proj_name}/${stage2_run_name}/${stage2_ckpt}"}

run_name="${exp_group}_stage3_${model_config}_${train_config}_${comment}"

cmd="accelerate launch"
cmd+=" --config_file=configs/accelerate/multigpu_config.yaml"
cmd+=" --num_machines=$n_machines"
cmd+=" --num_processes=$n_gpus"
cmd+=" --machine_rank=$machine_rank"
cmd+=" --main_process_port=$master_port"
cmd+=" stage3.py"
cmd+=" --init_from=${path_stage2}"
cmd+=" --init_method=build_from_stage2"
cmd+=" --proj_name=${proj_name}"
cmd+=" --model_name=${model_name}"
cmd+=" --run_name=${run_name}"
cmd+=" --tok_path=${tok_path}"
cmd+=" --stage=3"
cmd+=" --train_config=configs/training/${train_config}.json"
cmd+=" --model_config=configs/model/hypenet/${model_config}.json"

echo "==== Final command ===="
echo "$cmd"
echo "======================="
eval "$cmd"

echo "Stage 3 completed."
