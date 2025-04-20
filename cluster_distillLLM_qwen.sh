#!/bin/bash
#SBATCH -t 7:00:00  # time requested in hour:minute:second
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --constraint=a6000 #24 a6000, v100, a5000
#SBATCH --partition=compsci-gpu
#SBATCH --output=slurm_%j.out

srun hostname
srun date
# Define a writable directory for the virtual environment
export VENV_DIR=$HOME/final_project_distillLLM/venv
#mkdir -p $VENV_DIR
export HF_HOME=/dev/shm/hf-home
export TRANSFORMERS_CACHE=/dev/shm/hf-cache
export HF_DATASETS_CACHE=/dev/shm/hf-datasets
export TORCH_HOME=/dev/shm/torch-home
export XDG_CACHE_HOME=/dev/shm/.cache
export WANDB_CACHE_DIR=/dev/shm/wandb-cache

# 1. Perform a SFT on the Qwen2.5-1.5B teacher model
# srun bash -c "source \$HOME/final_project_distillLLM/minillm/.env && source \$VENV_DIR/bin/activate && huggingface-cli login --token \$HF_TOKEN && bash scripts/qwen2.5/sft/sft_1.5B.sh"

# 2. Perform a KD on the Qwen2.5-0.5B student model using the Qwen2.5-1.5B teacher model
srun bash -c "source \$HOME/final_project_distillLLM/minillm/.env && source \$VENV_DIR/bin/activate && huggingface-cli login --token \$HF_TOKEN && bash scripts/qwen2.5/kd/kd_1.5B_0.5B.sh"

# sbatch submit_llm.sh && JID=`squeu -u $USER -h -o%A` && sleep 5 && head slurm=$JID.out --lines=25