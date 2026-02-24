import os
import time
import random
import numpy as np
from datetime import timedelta
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import get_rank, group

import deepspeed
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    mpu
)

import time
import os

import torch
import torch.distributed as dist
import deepspeed

import json

from transformers import mpu

from arguments import get_args

from utils import initialize, print_args
from utils import print_rank
from utils import save_rank
from utils import get_tokenizer, get_model

from evaluate_main import evaluate_main, prepare_dataset_main
from evaluate_exposure_bias import evaluate_eb, prepare_dataset_eb

device = torch.cuda.current_device()

args = get_args()
initialize(args)

model_directory = args.model_path  #'./results/gpt2/train/sft/SFT-gpt2-120M'


tokenizer = AutoTokenizer.from_pretrained(model_directory)
# if args.model_type in ["gpt2", "opt", "llama", "gptj", "llama2", "mistral", "qwen2"]:
tokenizer.pad_token_id = tokenizer.eos_token_id

config = AutoConfig.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory, config=config).to('cuda')

dataset = prepare_dataset_main(
    args,
    tokenizer,
)

evaluate_main(args, tokenizer, model, dataset["test"], "test", 0, device)

del tokenizer, config, model