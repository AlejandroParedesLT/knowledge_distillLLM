base_path=${1-"."}
port=2040


for data in dolly # self_inst vicuna sinst uinst 
do
    # Evaluate SFT
    for seed in 10 20 30 40 50
    do
        ckpt="sft/SFT-gpt2-120M"
        bash ${base_path}/scripts/gpt2/eval/eval_main_${data}.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 8
    done

    # # Evaluate KD
    for seed in 10 20 30 40 50
    do
        ckpt="kd/KD-gpt2-120M"
        bash ${base_path}/scripts/gpt2/eval/eval_main_${data}.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 8
    done

    # # Evaluate SeqKD
    for seed in 10 20 30 40 50
    do
        ckpt="seqkd/SeqKD-gpt2-120M"
        bash ${base_path}/scripts/gpt2/eval/eval_main_${data}.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 8
    done

    # # Evaluate MiniLLM
    for seed in 10 20 30 40 50
    do
        ckpt="minillm/MiniLLM-gpt2-120M"
        bash ${base_path}/scripts/gpt2/eval/eval_main_${data}.sh ${base_path} ${port} 1 ${ckpt} --seed $seed  --eval-batch-size 8
    done
done