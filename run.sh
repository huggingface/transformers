export model_type="roberta"
export model="roberta-large"
# NOTE: USE --do_lower_case if using an uncased model (e.g. bert-base-uncased) !!!
export mode=${1:evaluate}
export gpuid=${3:-0}
export seed=${4:-1}

function train() {
    python examples/run_superglue.py --data_dir ${data_dir} --task_name ${task} \
                                     --output_dir ${out_dir} --overwrite_output_dir \
                                     --model_type ${model_type} --model_name_or_path ${model} \
                                     --use_gpuid ${gpuid} --seed ${seed} \
                                     --do_train --num_train_epochs 10 \
                                     --do_eval --evaluate_steps ${eval_freq} \
                                     --learning_rate ${lr} \
                                     --warmup_ratio 0.06 \
                                     --weight_decay 0.01 \
                                     --per_gpu_train_batch_size 4 \
                                     --gradient_accumulation_steps ${grad_acc_steps} \
                                     --logging_steps 100

}

function evaluate() {
    python examples/run_superglue.py --data_dir ${data_dir} --task_name ${task} \
                                     --output_dir ${out_dir} --overwrite_output_dir \
                                     --model_type ${model_type} --model_name_or_path ${model} \
                                     --use_gpuid ${gpuid} --seed ${seed} \
                                     --do_eval --log_energy_consumption
}

function analyze() {
    python examples/compute_efficiency_info.py --log_dir ${out_dir}
}

function debug() {
    python -m pdb examples/run_superglue.py --data_dir ${data_dir} --task_name ${task} \
                                     --output_dir ${out_dir} --overwrite_output_dir \
                                     --model_type ${model_type} --model_name_or_path ${model} \
                                     --use_gpuid ${gpuid} --seed ${seed} \
                                     --do_train --num_train_epochs 1 \
                                     --do_eval \ #--log_evaluate_during_training \
                                     --learning_rate ${lr} \
                                     --warmup_ratio 0.06 \
                                     --weight_decay 0.01 \
                                     --per_gpu_train_batch_size 4 \
                                     --gradient_accumulation_steps ${grad_acc_steps} \
                                     --logging_steps 100
}

function boolq() { # 85.2
    export task="boolq"
    export data_dir="${PROC}/mtl-sentence-representations/BoolQ"
    export lr=0.00003
    export grad_acc_steps=8
    export eval_freq=294
}

function cb() { # 98.2/96.4 acc/f1
    export task="cb"
    export data_dir="${PROC}/mtl-sentence-representations/CB"
    export lr=0.00003
    export grad_acc_steps=4
    export eval_freq=15
}

function copa() { # bad results
    export task="copa"
    export data_dir="${PROC}/mtl-sentence-representations/COPA"
    export lr=0.00003
    export grad_acc_steps=2 # {1, 2, 4} for respective batch size of {4, 8, 16}
    export eval_freq=50     # {100, 50, 25}
}

function multirc() { # 80.1 / 48.8 F1/EM (LR 1e-5); 77.7/41.6 (LR 3e-6)
    export task="multirc"
    export data_dir="${PROC}/mtl-sentence-representations/MultiRC"
    export lr=0.00003
    export grad_acc_steps=8
    export eval_freq=851
}

function record() { # really slowly training...
    export task="record"
    export data_dir="${PROC}/mtl-sentence-representations/ReCoRD"
    export lr=0.00003
    export grad_acc_steps=8 
    export eval_freq=1000 # total steps (w/ bz 32): 561510 !
}

function rte() { # 85.9 acc
    export task="rte"
    export data_dir="${PROC}/mtl-sentence-representations/RTE"
    export lr=0.00003
    export grad_acc_steps=8
    export eval_freq=77
}

function wic() { # 71.3
    export task="wic"
    export data_dir="${PROC}/mtl-sentence-representations/WiC"
    export lr=0.00003
    export grad_acc_steps=8
    export eval_freq=77
}

function wsc() { # 70.6 (avg acc + f1, LR 1e-5); 70.6 LR 3e-6; 60.3 LR 1e-6
    export task="wsc"
    export data_dir="${PROC}/mtl-sentence-representations/WSC"
    export lr=0.00003
    export grad_acc_steps=4
    export eval_freq=17
}


if [ $2 == "boolq" ]; then
    boolq
elif [ $2 == "cb" ]; then
    cb
elif [ $2 == "copa" ]; then
    copa
elif [ $2 == "multirc" ]; then
    multirc
elif [ $2 == "record" ]; then
    record
elif [ $2 == "rte" ]; then
    rte
elif [ $2 == "wic" ]; then
    wic
elif [ $2 == "wsc" ]; then
    wsc
else
    echo "Task $2 not found"
fi
#export dt=`date '+%d%m_%H%M'`
#export out_dir="${CKPTS}/transformers/superglue/${model}/${task}/${dt}"
export out_dir="${CKPTS}/transformers/superglue/${model}/${task}"
mkdir -p ${out_dir}

if [ ${mode} == "train" ]; then
    train
elif [ ${mode} == "evaluate" ]; then
    evaluate
elif [ ${mode} == "analyze" ]; then
    analyze
elif [ ${mode} == "debug" ]; then
    debug
else
    echo "Command ${mode} not found"

fi
