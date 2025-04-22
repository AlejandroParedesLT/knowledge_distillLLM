from huggingface_hub import HfApi
import os
# Upload a folder to a Hugging Face Sp
# Import environment variables
from dotenv import load_dotenv
load_dotenv()

api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
api.upload_large_folder(
    folder_path="./results/gpt2",
    repo_id="alejandroparedeslatorre/gpt2_sft_models",
    repo_type="model",
)

api = HfApi()
# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
api.upload_large_folder(
    folder_path="./results/qwen2.5",
    repo_id="alejandroparedeslatorre/qwen_codefinetuned",
    repo_type="model",
)