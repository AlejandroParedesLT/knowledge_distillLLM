import os
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig
)

from arguments import get_args
from utils import initialize, print_args, print_rank, save_rank, get_tokenizer, get_model
from evaluate_main import evaluate_main, prepare_dataset_main
from evaluate_exposure_bias import evaluate_eb, prepare_dataset_eb

import os
from dotenv import load_dotenv
load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_directory = './results/qwen2.5/train/sft/e10-bs1-lr1e-05-G2-N4-NN1/4000'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_directory)
tokenizer.pad_token = tokenizer.eos_token  # Avoid warning and decoding issues

config = AutoConfig.from_pretrained(model_directory)
model = AutoModelForCausalLM.from_pretrained(model_directory, config=config).to(device)

# Load base model
base_model_directory = "Qwen/Qwen2.5-1.5B-Instruct"
base_tokenizer = AutoTokenizer.from_pretrained(base_model_directory, token=os.getenv("HF_TOKEN"))
base_tokenizer.pad_token = base_tokenizer.eos_token  # Avoid padding warning
base_model = AutoModelForCausalLM.from_pretrained(base_model_directory, token=os.getenv("HF_TOKEN")).to(device)
# Prompt
prompt = (
    """
    Write a for loop in Python that prints the numbers from 1 to 10.
    """
)



for i in range(5):
    # Tokenize with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    # Generate
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=250,
            do_sample=True,
            top_p=1,
            top_k=50,
            temperature=1,
            num_return_sequences=1
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print("\n=== MODEL RESPONSE ===\n")
        print(decoded)


    # Tokenize
    base_inputs = base_tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    # Generate
    base_model.eval()
    with torch.no_grad():
        base_outputs = base_model.generate(
            input_ids=base_inputs["input_ids"],
            attention_mask=base_inputs["attention_mask"],
            pad_token_id=base_tokenizer.pad_token_id,
            eos_token_id=base_tokenizer.eos_token_id,
            max_length=250,
            do_sample=True,
            top_p=1,
            top_k=50,
            temperature=1,
            num_return_sequences=1
        )
        base_decoded = base_tokenizer.decode(base_outputs[0], skip_special_tokens=True).strip()
        print("\n=== BASE base_model_directory RESPONSE ===\n")
        print(base_decoded)