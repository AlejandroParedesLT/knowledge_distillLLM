import multiprocessing
import os
import time
import torch
import json
import sys
import numpy as np
from data_utils.indexed_dataset import make_builder, best_fitting_dtype
from transformers import AutoTokenizer
from arguments import get_args


# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder(object): 
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

    def encode(self, line):
        try:
            # Try to load the line as JSON
            line = json.loads(line)
        except json.JSONDecodeError as e:
            # Log the error with a message and a snippet of the problematic line
            print(f"Skipping corrupted line due to JSON error: {e} in line: {line[:100]}...")
            return None, None, None, None, len(line)  # Return None to skip this line
        if "original_string" not in line or not line["original_string"]:
            raise ValueError("Key original_string is missing or empty in the input data.")
        else:
            if self.args.model_type not in ["qwen2"]:
                template = (
                    "Below is an instruction that describes a task. "
                    "Write a programming script that appropriately completes the request.\n\n"
                    "### Request:\n{summary}\n"
                    "### Response:\n"
                )
            else:
                template = (
                    "<|im_start|>Below is an instruction that describes a task. "
                    "Write a programming script that appropriately completes the request.\n\n"
                    "### Request:\n{summary}\n"
                    "### Response:\n\n<|im_end|>"
                )
            if "summary" in line:
                prompt = template.format(summary=line["summary"])
            else:
                # If "summary" is not present, use "docsting" as a fallback
                prompt = template.format(summary=line["docstring"])
            

        pass_code = '### solution code \n\n'+ line["original_string"] + "\n\n### End of code\n"
        prompt_tokens = Encoder.tokenizer.encode(prompt, add_special_tokens=False, truncation=True,)
        full_tokens = Encoder.tokenizer.encode(prompt + pass_code, add_special_tokens=False, truncation=True,) + [Encoder.tokenizer.eos_token_id]
        response_tokens = full_tokens[len(prompt_tokens):]
        
        if len(prompt_tokens) > self.args.max_prompt_length:
            return None, None, None, None, len(line)
        
        return line, prompt, prompt_tokens, response_tokens, len(line)


def main():
    print("OK")
    args = get_args()
        
    args.processed_data_dir = os.path.join(args.processed_data_dir, args.model_type)

    os.makedirs(args.processed_data_dir, exist_ok=True)
    
    # with open(os.path.join(args.data_dir, "core_trained_merged.jsonl")) as f:
    #     raw_data = f.readlines()

    # from datasets import load_dataset

    # dataset = load_dataset(
    #     "alejandroparedeslatorre/knowledgedistillation_training",
    #     data_files="core_trained_merged.jsonl"
    # )

    # raw_data = [json.dumps(example) for example in dataset["train"]]

    from huggingface_hub import hf_hub_download

    file_path = hf_hub_download(
        repo_id="alejandroparedeslatorre/knowledgedistillation_training",
        filename="core_trained_merged.jsonl",
        repo_type="dataset"
    )

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = f.readlines()

    if args.dev_num > 0:
        all_data = {
            "valid": raw_data[:args.dev_num],
            "train": raw_data[args.dev_num:]
        }
    else:
        all_data = {
            "train": raw_data
        }
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dtype = best_fitting_dtype(len(tokenizer))
    split_id = np.iinfo(dtype).max
    print("dtype:", dtype, "split_id:", split_id)
    for split in all_data:
        print("Split:", split)
        # encoder use the tokenizer to encode data
        encoder = Encoder(args)
        print("Encoder initialized.")
        # 2. Mapping all datas with Encoder, with the help of multiprocessing
        pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
        print("Start encoding data...")
        encoded_docs = pool.imap_unordered(encoder.encode, all_data[split], chunksize=50)
        print("Start processing data...")
        proc_start = time.time()
        total_bytes_processed = 0
        
        bin_file = os.path.join(args.processed_data_dir, f"{split}_{0}.bin")
        idx_file = os.path.join(args.processed_data_dir, f"{split}_{0}.idx")
        print("bin_file:", bin_file)
        print("idx_file:", idx_file)
        binary_builder = make_builder(bin_file, impl="mmap", dtype=dtype)
        print("Binary builder initialized.")
        # put tokenized data into binary_builder
        inst_num = 0
        print("#"*10, split, "#"*10)
        
        prompt_lens = []
        response_lens = []
        print("Prompt lengths:", prompt_lens)
        json_file = open(os.path.join(args.processed_data_dir, f"{split}.jsonl"), "w")
        print("JSON file opened.")
        for lid, (line, prompt_str, prompt, response, bytes_processed) in enumerate(encoded_docs):
            total_bytes_processed += bytes_processed
            if prompt is None:
                continue
            
            if args.only_prompt:
                if len(prompt) < args.max_length:
                    binary_builder.add_item(torch.IntTensor(prompt))
                else:
                    continue
            else:
                binary_builder.add_item(torch.IntTensor(prompt + [split_id] + response))

            json_file.write(json.dumps({
                "instruction": line["summary"] if "summary" in line else line["docstring"],
                "prompt": prompt_str,
                "output": line["original_string"],
            }) + "\n")

            prompt_lens.append(len(prompt))
            response_lens.append(len(response))

            inst_num += 1
            if lid % 1000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {lid} documents. {inst_num} instances.",
                    f"({lid/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)
            if lid % 500000 == 0 and lid > 0:
                print("Saving intermediate data...")
                break

        print("Processed", inst_num, "instances.")
        # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
        binary_builder.finalize(idx_file)
        print("Binary builder finalized.")

        # close multiproceessing mapping
        print("Closing pool...")
        pool.close()
        print("Pool closed.")
        # json_file.flush()
        # os.fsync(json_file.fileno())
        json_file.close()
                
        print("Data num", len(prompt_lens))
        print("Prompt lengths.", "Mean:", np.mean(prompt_lens), "Max:", np.max(prompt_lens), "Min:", np.min(prompt_lens))
        print("Response", "Mean:", np.mean(response_lens), "Max:", np.max(response_lens), "Min:", np.min(response_lens))
    print("All data processed.")
    sys.exit(0)
    print("Confirmed exit.")

if __name__ == '__main__':
    main()
    