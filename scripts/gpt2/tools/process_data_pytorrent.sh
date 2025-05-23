#!/bin/bash
set -e
set -x  # Print commands as they execute (helps debugging)

BASE_PATH=${1}
export TF_CPP_MIN_LOG_LEVEL=3
# echo ">>> Running prompt-only"
# # only prompt for MiniLLM train
# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pytorrent.py \
#     --data-dir ${BASE_PATH}/data/pytorrent/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/pytorrent/prompt \
#     --model-path gpt2-xl \
#     --data-process-workers 32 \
#     --max-prompt-length 256 \
#     --dev-num 1000 \
#     --only-prompt \
#     --model-type gpt2
python3 -c "print('Hello'); import time; time.sleep(1)"

echo ">>> Running prompt+response"
# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pytorrent.py \
    --data-dir ${BASE_PATH}/data/pytorrent/ \
    --processed-data-dir ${BASE_PATH}/processed_data/pytorrent/full \
    --model-path gpt2-xl \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type gpt2