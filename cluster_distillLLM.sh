#!/bin/bash
#SBATCH -t 5:00:00  # time requested in hour:minute:second
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a5000 #24 a6000, v100, a5000
#SBATCH --partition=compsci-gpu
#SBATCH --output=slurm_%j.out

srun hostname
srun date
# Define a writable directory for the virtual environment
export VENV_DIR=$HOME/final_project_distillLLM/venv
#mkdir -p $VENV_DIR

#srun python3 -m venv $VENV_DIR
# srun bash -c "source $VENV_DIR/bin/activate && bash install.sh"
# srun bash -c "nvidia-smi"

#srun bash -c "source $VENV_DIR/bin/activate && bash scripts/gpt2/tools/process_data_dolly.sh ."
#srun bash -c "source $VENV_DIR/bin/activate && bash scripts/gpt2/tools/process_data_pretrain.sh ."
#srun bash -c "source $VENV_DIR/bin/activate && bash scripts/gpt2/eval/run_eval.sh"

srun bash -c "source $VENV_DIR/bin/activate && bash scripts/gpt2/sft/sft_xlarge_bugnet.sh"

#srun bash -c "source $VENV_DIR/bin/activate && bash scripts/gpt2/sft/sft_base_bugnet.sh"


#srun bash -c "source $VENV_DIR/bin/activate && bash python hello.py ."

# sbatch submit_llm.sh && JID=`squeu -u $USER -h -o%A` && sleep 5 && head slurm=$JID.out --lines=25