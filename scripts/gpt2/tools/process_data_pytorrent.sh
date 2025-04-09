BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

# # only prompt for MiniLLM train
# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pytorrent.py \
#     --data-dir ${BASE_PATH}/data/pytorrent/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/pytorrent/prompt \
#     --model-path ${BASE_PATH}/checkpoints/gpt2-xlarge \
#     --data-process-workers 32 \
#     --max-prompt-length 256 \
#     --dev-num 1000 \
#     --only-prompt \
#     --model-type gpt2

# # prompt and response for baselines
# PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pytorrent.py \
#     --data-dir ${BASE_PATH}/data/pytorrent/ \
#     --processed-data-dir ${BASE_PATH}/processed_data/pytorrent/full \
#     --model-path ${BASE_PATH}/checkpoints/gpt2-xlarge \
#     --data-process-workers 32 \
#     --max-prompt-length 256 \
#     --dev-num 1000 \
#     --model-type gpt2


# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pytorrent.py \
    --data-dir ${BASE_PATH}/data/pytorrent/ \
    --processed-data-dir ${BASE_PATH}/processed_data/pytorrent/prompt \
    --model-path gpt2-xl \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --only-prompt \
    --model-type gpt2

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pytorrent.py \
    --data-dir ${BASE_PATH}/data/pytorrent/ \
    --processed-data-dir ${BASE_PATH}/processed_data/pytorrent/full \
    --model-path gpt2-xl \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type gpt2
