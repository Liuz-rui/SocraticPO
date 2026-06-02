#!/bin/bash
#SBATCH -J qwen3_tg
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH -w gpu02

set -xeuo pipefail

cd /data/tingyue/workspace/Agent-R1
mkdir -p logs
ml miniconda/26.1.1
ml cuda/12.4

eval "$(conda shell.bash hook)"
conda activate verl070

export LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH

unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export RAY_ADDRESS=local
export SWANLAB_MODE=local

export VLLM_USE_V1=1
export HF_ENDPOINT=https://hf-mirror.com

# ==========================================
# 教师引导与外部接口配置
# ==========================================
export TEACHER_GUIDANCE_MODE="teacher"
MAX_STEPS=${MAX_STEPS:-2}
TRAIN_GPUS_PER_NODE=4

export TEACHER_MODEL_NAME="Qwen3-4B-Instruct-2507"
export TEACHER_API_KEY=empty
export TEACHER_ENDPOINT=${TEACHER_ENDPOINT:-"http://172.16.177.1:8002/v1"}

# ==========================================
# 检查外部教师服务是否连通
# ==========================================
if [[ "$MAX_STEPS" -gt 1 ]]; then
  echo "Checking external teacher at ${TEACHER_ENDPOINT}..."
  if ! curl -sf "${TEACHER_ENDPOINT}/models" >/dev/null; then
    echo "Warning: Cannot connect to external teacher at ${TEACHER_ENDPOINT}. Ensure the service is running!" >&2
  else
    echo "External teacher is reachable."
  fi
fi

# ==========================================
# 实验参数与启动
# ==========================================
DATASET_TAG=${DATASET_TAG:-biology}
DATASET_WITH_SOLUTION=${DATASET_WITH_SOLUTION:-False}
case "$DATASET_WITH_SOLUTION" in
  True|true|1|yes|Yes) DEFAULT_DATASET_ROOT="$HOME/data/sciknoweval" ;;
  False|false|0|no|No) DEFAULT_DATASET_ROOT="$HOME/data/sciknoweval_without_solution" ;;
  *) echo "Invalid DATASET_WITH_SOLUTION" >&2; exit 1 ;;
esac

DATASET_ROOT=${DATASET_ROOT:-"$DEFAULT_DATASET_ROOT"}
TRAIN_FILE=${TRAIN_FILE:-"${DATASET_ROOT}/${DATASET_TAG}/train.parquet"}
VAL_FILE=${VAL_FILE:-"${DATASET_ROOT}/${DATASET_TAG}/test.parquet"}
MODEL_PATH=${MODEL_PATH:-"/data/tingyue/.cache/modelscope/hub/models/Qwen/Qwen3-4B-Instruct-2507"}

PROJECT_NAME=${PROJECT_NAME:-"tg_qwen3_4b"}
RUN_ID=${SLURM_JOB_ID:-local}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-reinforce_plus_plus_baseline}
GROUP_ADVANTAGE_BY_STEP=${GROUP_ADVANTAGE_BY_STEP:-False}
PRIOR_STEP_MEAN_REWARD_SCALE=${PRIOR_STEP_MEAN_REWARD_SCALE:-False}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-1e-6}
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_N=${VAL_N:-16}

case "$ADV_ESTIMATOR" in
  reinforce_plus_plus_baseline) ADV_ESTIMATOR_TAG=rppb ;;
  grpo) ADV_ESTIMATOR_TAG=grpo ;;
  reinforce_plus_plus) ADV_ESTIMATOR_TAG=rpp ;;
  reinforce) ADV_ESTIMATOR_TAG=reinf ;;
  *) ADV_ESTIMATOR_TAG=$ADV_ESTIMATOR ;;
esac

GROUP_ADVANTAGE_BY_STEP_TAG=$([[ "$GROUP_ADVANTAGE_BY_STEP" =~ ^(True|true)$ ]] && echo "1" || echo "0")
GUIDANCE_MODE_TAG=$([[ "$MAX_STEPS" -le 1 ]] && echo "base" || echo "tg")

EXPERIMENT_NAME=${EXPERIMENT_NAME:-"q34_${GUIDANCE_MODE_TAG}_${DATASET_TAG}_${ADV_ESTIMATOR_TAG}_gs${GROUP_ADVANTAGE_BY_STEP_TAG}_n${ROLLOUT_N}_vn${VAL_N}_s${MAX_STEPS}_bs${TRAIN_BATCH_SIZE}_lr${LEARNING_RATE}_${RUN_ID}"}

python3 -m agent_r1.main_agent_ppo \
  algorithm.adv_estimator=$ADV_ESTIMATOR \
  +algorithm.use_verl_advantage=True \
  +algorithm.group_advantage_by_step=$GROUP_ADVANTAGE_BY_STEP \
  +algorithm.prior_step_mean_reward_scale=$PRIOR_STEP_MEAN_REWARD_SCALE \
  data.train_files=$TRAIN_FILE \
  data.val_files=$VAL_FILE \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.max_prompt_length=8192 \
  data.max_response_length=8192 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=$VAL_N \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.prompt_length=8192 \
  actor_rollout_ref.rollout.response_length=8192 \
  actor_rollout_ref.rollout.max_model_len=16384 \
  actor_rollout_ref.rollout.agent.default_agent_flow=teacher_guidance_agent \
  actor_rollout_ref.rollout.agent.agent_flow_config_path=examples/teacher_guidance_agent.yaml \
  actor_rollout_ref.rollout.agent.max_steps=$MAX_STEPS \
  critic.enable=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.logger='["console","swanlab"]' \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node=$TRAIN_GPUS_PER_NODE \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=20 \
  trainer.total_epochs=200 \
