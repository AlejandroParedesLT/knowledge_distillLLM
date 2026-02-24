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
import gc

device = torch.cuda.current_device()

args = get_args()
initialize(args)

model_directory='./results/gpt2/train/sft/e10-bs2-lr1e-05-G1-N1-NN1/20' #args.model_path  #'./results/gpt2/train/sft/SFT-gpt2-120M'


tokenizer = AutoTokenizer.from_pretrained(model_directory)
# if args.model_type in ["gpt2", "opt", "llama", "gptj", "llama2", "mistral", "qwen2"]:
tokenizer.pad_token_id = tokenizer.eos_token_id

config = AutoConfig.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory, config=config).to('cuda:0')

model.generate()


# Load model configuration and model
config = AutoConfig.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory, config=config).to(device)
model.eval()  # Set model to evaluation mode

def generate_text(prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.95):
    """
    Generate text based on a given prompt using the trained model.
    
    Parameters:
    - prompt (str): Input text prompt.
    - max_length (int): Maximum length of the generated sequence.
    - temperature (float): Sampling temperature (higher values = more randomness).
    - top_k (int): Top-k filtering (0 to disable).
    - top_p (float): Nucleus sampling probability (0-1).
    
    Returns:
    - generated_text (str): The generated text.
    """
    # Tokenize input prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode output tokens to text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def cleanup():
    """Deletes the model and clears GPU cache."""
    global model, tokenizer
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Model deleted and GPU cache cleared.")

if __name__ == "__main__":
    prompt = input("Enter a prompt: ")
    generated_text = generate_text(prompt)
    cleanup()
    print("\nGenerated Text:\n", generated_text)