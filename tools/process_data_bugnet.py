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
        line = json.loads(line)
        # Check if "fail" exists and is non-empty
        if "fail" not in line or not line["fail"]:
            raise ValueError("Key 'fail' is missing or empty in the input data.")
        else:
            if self.args.model_type not in ["qwen2"]:
                template = (
                    "Below is an instruction that describes a task. "
                    "Write a C++ Script that contains the correction of the code bug in order to fix the bug, fix the issue or fix the error\n\n"
                    "### Code that has the bug issue or is incorrect:\n{fail}\n"
                    "### The reported error in the code:\n{error}\n\n"
                    "### Original Status:\n{original_status}\n\n"
                    "### Response:\n"
                )
            else:
                template = (
                    "<|im_start|>Below is an instruction that describes a task. "
                    "Write a Python Script that contains the correction of the code bug in order to fix the bug, fix the issue or fix the error\n\n"
                    "### Code that has the bug issue or is incorrect:\n{fail}\n"
                    "### Use as context the reported error in the code:\n{error}\n\n### Response:\n<|im_end|><|im_start|>Assistant:"
                )
            prompt = template.format(fail=line["fail"],error=line["error"],original_status=line["original_status"])
            
        if line["change"]:
            pass_code = 'Solution type '+line["change"]+', solution code'+line["pass"]
        else:
            pass_code = 'Solution type undetermined, solution code'+ line["pass"]
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
    
    with open(os.path.join(args.data_dir, "cpp_trainvalid.jsonl")) as f:
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
        
        # encoder use the tokenizer to encode data
        encoder = Encoder(args)

        # 2. Mapping all datas with Encoder, with the help of multiprocessing
        pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
        encoded_docs = pool.imap_unordered(encoder.encode, all_data[split], chunksize=50)
        proc_start = time.time()
        total_bytes_processed = 0
        
        bin_file = os.path.join(args.processed_data_dir, f"{split}_{0}.bin")
        idx_file = os.path.join(args.processed_data_dir, f"{split}_{0}.idx")

        binary_builder = make_builder(bin_file, impl="mmap", dtype=dtype)

        # put tokenized data into binary_builder
        inst_num = 0
        print("#"*10, split, "#"*10)
        
        prompt_lens = []
        response_lens = []
        
        json_file = open(os.path.join(args.processed_data_dir, f"{split}.jsonl"), "w")
        
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
                "fail": line["fail"],
                "prompt": prompt_str,
                "error": line["error"],
                "original_status": line["original_status"],
                "output": line["pass"],
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

        # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
        binary_builder.finalize(idx_file)

        # close multiproceessing mapping
        pool.close()
        json_file.close()
                
        print("Data num", len(prompt_lens))
        print("Prompt lengths.", "Mean:", np.mean(prompt_lens), "Max:", np.max(prompt_lens), "Min:", np.min(prompt_lens))
        print("Response", "Mean:", np.mean(response_lens), "Max:", np.max(response_lens), "Min:", np.min(response_lens))


if __name__ == '__main__':
    main()