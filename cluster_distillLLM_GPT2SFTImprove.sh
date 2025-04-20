#!/bin/bash
#SBATCH -t 15:00:00  # time requested in hour:minute:second
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --constraint=a5000 #24 a6000, v100, a5000
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

#srun python3 -m venv $VENV_DIR
# srun bash -c "source $VENV_DIR/bin/activate && bash install.sh"
# srun bash -c "nvidia-smi"

#srun bash -c "source $VENV_DIR/bin/activate && bash scripts/gpt2/tools/process_data_dolly.sh ."
#srun bash -c "source $VENV_DIR/bin/activate && bash scripts/gpt2/tools/process_data_pretrain.sh ."
#srun bash -c "source $VENV_DIR/bin/activate && bash scripts/gpt2/eval/run_eval.sh"

# Running successfully
#srun bash -c "source $VENV_DIR/bin/activate && bash scripts/gpt2/sft/sft_xlarge_bugnet.sh"
#srun bash -c "source $VENV_DIR/bin/activate && bash scripts/gpt2/sft/sft_base_bugnet.sh"
#srun bash -c "source $VENV_DIR/bin/activate && bash scripts/gpt2/sft/sft_xlarge_dollySpanish.sh"

srun bash -c "source \$HOME/final_project_distillLLM/minillm/.env && source \$VENV_DIR/bin/activate && huggingface-cli login --token \$HF_TOKEN && bash scripts/gpt2/sft/sft_xlarge_pytorrent.sh"

#srun bash -c "source $VENV_DIR/bin/activate && bash scripts/gpt2/tools/generate_data_seqkd_bugnet.sh ."
#srun bash -c "source $VENV_DIR/bin/activate && bash python hello.py ."

# sbatch submit_llm.sh && JID=`squeu -u $USER -h -o%A` && sleep 5 && head slurm=$JID.out --lines=25