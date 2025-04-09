# PyTorrent: Knowledge Distillation Framework for LLMs

## Overview

PyTorrent is a framework for fine-tuning and knowledge distillation of large language models. It supports supervised fine-tuning (SFT) and knowledge distillation (KD) techniques for training models like GPT-2 on Python code datasets.

## Requirements

- Python 3.8+
- PyTorch
- DeepSpeed
- Transformers
- Accelerate
- PEFT (Parameter-Efficient Fine-Tuning)

## Project Structure

- `finetune.py`: Main training script that handles SFT and KD
- Training scripts:
  - `sft_xlarge_pytorrent.sh`: Script for supervised fine-tuning on GPT2-XL
  - `kd_base_bugnet.sh`: Basic knowledge distillation script 
  - `seqkd_base_bugnet.sh`: Sequence knowledge distillation script
- Evaluation:
  - `alejandro_eval.py`: Evaluation script for trained models

## Features

- Supervised fine-tuning (SFT) of language models
- Knowledge distillation (KD) from larger teacher models to smaller student models
- DeepSpeed integration for efficient training across multiple GPUs
- Support for model parallelism
- PEFT techniques (e.g., LoRA) compatibility
- Flexible evaluation with metrics like ROUGE

## Usage

### Supervised Fine-Tuning

To fine-tune a GPT2-XL model on the PyTorrent dataset:

```bash
./sft_xlarge_pytorrent.sh [BASE_PATH] [PORT]
```

### Knowledge Distillation

To distill knowledge from a fine-tuned GPT2-XL model to a smaller GPT2 model:

```bash
./kd_base_bugnet.sh [BASE_PATH] [PORT]
```

For sequence-level knowledge distillation:

```bash
./seqkd_base_bugnet.sh [BASE_PATH] [PORT]
```

### Evaluation

To evaluate a trained model:

```bash
python alejandro_eval.py --model-path [MODEL_PATH]
```

## Configuration

The framework uses DeepSpeed configurations for distributed training. Configuration files are expected in `configs/deepspeed/`.

Each training script includes various hyperparameters:
- Learning rate
- Batch size
- Gradient accumulation steps
- Epochs
- KD ratio (for knowledge distillation)
- Maximum sequence length
- Generation parameters (top-k, top-p, temperature)

## Datasets

The framework is configured to work with:
- PyTorrent: Python code dataset
- BugNet: Python code with bug fixes

Data should be processed and placed in the appropriate directory structure:
```
processed_data/
├── pytorrent/
│   └── full/
│       └── gpt2/
└── bugnet_python/
    └── full/
        └── gpt2/
```

This repo was based on the following research study:

@inproceedings{minillm,
  title={MiniLLM: Knowledge Distillation of Large Language Models},
  author={Gu, Yuxian and Dong, Li and Wei, Furu and Huang, Minlie},
  booktitle={Proceedings of ICLR},
  year={2024}
}
