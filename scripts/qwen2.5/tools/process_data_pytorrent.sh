BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# # only prompt for MiniLLM train
# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pytorrent.py \
#     --data-dir ${BASE_PATH}/data/dolly/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/pytorrent/prompt \
#     --model-path Qwen/Qwen2.5-1.5B-Instruct \
#     --data-process-workers 32 \
#     --max-prompt-length 256 \
#     --dev-num 1000 \
#     --only-prompt \
#     --model-type qwen2
python3 -c "print('Hello'); import time; time.sleep(1)"
# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pytorrent.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/pytorrent/full \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type qwen2
