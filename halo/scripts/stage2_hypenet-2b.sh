args="$@"
for arg in "$@"; do
    eval "$arg"
done

gpus_per_node=${gpus_per_node:-8}
n_machines=${n_machines:-1}
machine_rank=${machine_rank:-0}
n_gpus=$(( $n_machines * $gpus_per_node ))

proj_name=${proj_name:-"hypenet"}
exp_group=${exp_group:-"e1"}
model_name=${model_name:-"LA"}
train_config=${train_config:-"stage2"}
model_config=${model_config:-"hypenet-2b"}
orig_model=${orig_model:-"/path/to/qwen3-1.7b"}
comment=${comment:-""}

stage1_run_name=${stage1_run_name:-"${exp_group}_${model_config}_stage1_"}
stage1_ckpt=${stage1_ckpt:-"ckpt_20000"}

path_stage1=${path_stage1:-"results/${proj_name}/${stage1_run_name}/${stage1_ckpt}"}
run_name="${exp_group}_stage2_${model_config}_${train_config}_${comment}"

cmd="accelerate launch"
cmd+=" --config_file=configs/accelerate/multigpu_config.yaml"
cmd+=" --num_machines=$n_machines"
cmd+=" --num_processes=$n_gpus"
cmd+=" --machine_rank=$machine_rank"
cmd+=" stage2.py"
# cmd+=" --use_deepspeed=1"
cmd+=" --init_from=${path_stage1}"
cmd+=" --proj_name=${proj_name}"
cmd+=" --model_name=${model_name}"
cmd+=" --run_name=${run_name}"
cmd+=" --tok_path=${orig_model}"
cmd+=" --teacher_model=${orig_model}"
cmd+=" --stage=2"
cmd+=" --train_config=configs/training/${train_config}.json"
cmd+=" --model_config=configs/model/hypenet/${model_config}.json"


echo "==== Final command ===="
echo "$cmd"
echo "======================="
eval "$cmd"

echo "Stage 2 completed."
