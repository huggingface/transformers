

#for task in copa wsc; do
function copa() {
    export task="copa"
    for lr in .00001 .00002 .00003; do
        for bz in 1 2 4; do
            for seed in 1 2 3 4 5; do
                job_name=${task}-lr${lr}-bz${bz}-seed${seed} task=${task} seed=${seed} lr=${lr} bz=${bz} sbatch nyu_prince_aw.sbatch
            done
        done
    done
}

function multirc() {
    export task="multirc"
    export bz=8
    #export lr=.00003
    export seed=1
    #for seed in 1 2 3; do
    for lr in .00002 .00001 .000005; do
        job_name=${task}-lr${lr}-bz${bz}-seed${seed} task=${task} seed=${seed} lr=${lr} bz=${bz} sbatch nyu_prince_aw.sbatch
    done
}

function record() {
    export task="record"
    export lr=.00003
    export bz=8
    for seed in 1 2 3; do
        job_name=${task}-lr${lr}-bz${bz}-seed${seed} task=${task} seed=${seed} lr=${lr} bz=${bz} sbatch nyu_prince_aw.sbatch
    done
}

function wsc() {
    export task="wsc"
    for lr in .00001 .00002 .00003; do
        for bz in 1 2 4; do
            for seed in 1 2 3 4 5; do
                job_name=${task}-lr${lr}-bz${bz}-seed${seed} task=${task} seed=${seed} lr=${lr} bz=${bz} sbatch nyu_prince_aw.sbatch
            done
        done
    done
}

function debug() {
    export task="wsc"
    for lr in .00001; do
        for bz in 1; do
            for seed in 1; do
                job_name=${task}-lr${lr}-bz${bz}-seed${seed} task=${task} seed=${seed} lr=${lr} bz=${bz} sbatch nyu_prince_aw.sbatch
            done
        done
    done
}

echo "MAKE SURE TO ADJUST JOB LENGTH AND MACHINE!!!"
sleep 2
if [ $1 == "debug" ]; then
    debug
elif [ $1 == "copa" ]; then
    copa
elif [ $1 == "wsc" ]; then
    wsc
elif [ $1 == "record" ]; then
    record
elif [ $1 == "multirc" ]; then
    multirc
else
    echo "Unknown command ${1}"
fi
