#!/bin/bash
#SBATCH --job-name=build_nemo
#SBATCH --partition=compsci-gpu
#SBATCH --mem=128G
#SBATCH --gres=gpu:0
#SBATCH --time=01:00:00
#SBATCH --output=build_nemo.log

# Set up tmpdir to RAM
export TMPDIR=/dev/shm
export APPTAINER_CACHEDIR=/dev/shm/apptainer_cache

torchrun --nproc_per_node 8 scripts/llm/gpt_prune.py \
    --devices 8 \
    --tp_size 1 \
    --pp_size 8 \
    --restore_path <path/to/llama3.1-8b-nemo2> \
    --seq_length 8192 \
    --data_paths 30 path/to/dataset_1_prefix 70 path/to/dataset_2_prefix \
    --index_mapping_dir path/to/index_mapping_dir \
    --target_ffn_hidden_size 9216 \
    --target_hidden_size 3072 \
    --target_num_attention_heads 32 \
    --target_num_query_groups 8 \
    --target_num_layers 16 \
    --save_path llama3.1-8b-width-depth-pruned