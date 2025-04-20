#!/bin/bash
#SBATCH -t 1:00:00  # time requested in hour:minute:second
#SBATCH --job-name=build_nemo
#SBATCH --partition=compsci-gpu
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a5000
#SBATCH --output=slurm_%j.out

# Set up tmpdir to RAM
export TMPDIR=/dev/shm
export APPTAINER_CACHEDIR=/dev/shm/apptainer_cache

# Change to RAM-backed directory
cd /dev/shm

echo "Pulling image into RAM..."
apptainer pull NvidiaPytorchContainer.sif docker://nvcr.io/nvidia/pytorch:25.03-py3

echo "Building NeMo environment inside the container..."
apptainer exec --nv NvidiaPytorchContainer.sif bash -c "
    # Create and activate Conda environment
    conda create --yes --name nemo python=3.10.12
    source activate nemo

    # Install PyTorch and related packages
    conda install --yes pytorch=2.2.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

    # Install system dependencies
    apt-get update && apt-get install -y libsndfile1 ffmpeg

    # Install NeMo toolkit
    pip install Cython packaging
    pip install nemo_toolkit[all]
"

echo "Copying the built SIF image from RAM to local storage..."
cp /dev/shm/NvidiaPytorchContainer.sif .apptainer/NvidiaPytorchContainer.sif
