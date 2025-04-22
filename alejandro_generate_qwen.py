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

###################
# Base model
base_model_directory = "Qwen/Qwen2.5-0.5B"
base_tokenizer = AutoTokenizer.from_pretrained(base_model_directory, token=os.getenv("HF_TOKEN"))
base_tokenizer.pad_token = base_tokenizer.eos_token  # Avoid padding warning
base_model = AutoModelForCausalLM.from_pretrained(base_model_directory, token=os.getenv("HF_TOKEN")).to(device)

#####################
# TEacher Model
teacher_model_directory = './results/qwen2.5/train/sft/qwen2.5-1.5B-Instruct/e10-bs1-lr1e-05-G2-N4-NN1/8000'
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(teacher_model_directory)
tokenizer.pad_token = tokenizer.eos_token  # Avoid warning and decoding issues
config = AutoConfig.from_pretrained(teacher_model_directory)
model = AutoModelForCausalLM.from_pretrained(teacher_model_directory, config=config).to(device)

#######################
# Student model SFT
base_sft_model_directory = './results/qwen2.5/train/sft/qwen2.5-0.5B-Instruct/e10-bs1-lr1e-05-G2-N2-NN1/8000'
# Load tokenizer and model
tokenizer_sft = AutoTokenizer.from_pretrained(base_sft_model_directory)
tokenizer_sft.pad_token = tokenizer_sft.eos_token  # Avoid warning and decoding issues
config = AutoConfig.from_pretrained(base_sft_model_directory)
model_sft = AutoModelForCausalLM.from_pretrained(base_sft_model_directory, config=config).to(device)

# ./results/qwen2.5/train/sft/qwen2.5-0.5B-Instruct/e10-bs1-lr1e-05-G2-N2-NN1/8000

###################
# Student Model
student_model_directory = './results/qwen2.5/train/kd/Qwen2.5-0.5B-to-Qwen2.5-1.5B-sft/e10-bs8-lr1e-05-G1-N2-NN1-kd0.5/8000'

# Load tokenizer and model
tokenizer_student = AutoTokenizer.from_pretrained(student_model_directory)
tokenizer_student.pad_token = tokenizer_student.eos_token  # Avoid warning and decoding issues
config_student = AutoConfig.from_pretrained(student_model_directory)
model_student = AutoModelForCausalLM.from_pretrained(student_model_directory, config=config_student).to(device)



prompt = (
    """
    How can I implement depth first search in python?
    """
)



for i in range(1):
    # Tokenize with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

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
        # Write results to a .txt file
        with open("responses_qwen.txt", "a") as f:
            f.write(f"############## BASE MODEL ##############\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Response: {base_decoded}\n")
            f.write("="*50 + "\n")


    # Generate
    model_sft.eval()
    with torch.no_grad():
        outputs = model_sft.generate(
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
        # Write results to a .txt file
        with open("responses_qwen.txt", "a") as f:
            f.write(f"############## STUDENT SFT MODEL ##############\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Response: {decoded}\n")
            f.write("="*50 + "\n")

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
        # Write results to a .txt file
        with open("responses_qwen.txt", "a") as f:
            f.write(f"############## TEACHER MODEL ##############\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Response: {decoded}\n")
            f.write("="*50 + "\n")
    
    # Student Model
    # Tokenize with attention massk
    inputs_student = tokenizer_student(prompt, return_tensors="pt", padding=True).to(device)
    # Generate
    model_student.eval()
    with torch.no_grad():
        outputs_student = model_student.generate(
            input_ids=inputs_student["input_ids"],
            attention_mask=inputs_student["attention_mask"],
            pad_token_id=tokenizer_student.pad_token_id,
            eos_token_id=tokenizer_student.eos_token_id,
            max_length=250,
            do_sample=True,
            top_p=1,
            top_k=50,
            temperature=1,
            num_return_sequences=1
        )
        decoded_student = tokenizer_student.decode(outputs_student[0], skip_special_tokens=True).strip()
        print("\n=== STUDENT MODEL RESPONSE ===\n")
        print(decoded_student)
        # Write results to a .txt file
        with open("responses_qwen.txt", "a") as f:
            f.write(f"############## STUDENT MODEL ##############\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Response: {decoded_student}\n")
            f.write("="*50 + "\n")


