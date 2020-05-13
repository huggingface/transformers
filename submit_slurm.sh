
function boolq() {
    export task="boolq"
    for lr in .00003; do
        for nagg in 8; do
            for seed in 1 2 3 4 5; do
                job_name=${task}-lr${lr}-nagg${nagg}-seed${seed} task=${task} seed=${seed} lr=${lr} nagg=${nagg} sbatch nyu_cassio_aw.sbatch
            done
        done
    done
}

function cb() {
    export task="cb"
    for lr in .00003; do
        for nagg in 4; do
            for seed in 1 2 3 4 5; do
                job_name=${task}-lr${lr}-nagg${nagg}-seed${seed} task=${task} seed=${seed} lr=${lr} nagg=${nagg} sbatch nyu_cassio_aw.sbatch
            done
        done
    done
}

function copa() {
    export task="copa"
    for lr in .000003 .000005 .00001; do
        for nagg in 1 2 4; do
            for seed in 1 2 3 4 5; do
                job_name=${task}-lr${lr}-nagg${nagg}-seed${seed} task=${task} seed=${seed} lr=${lr} nagg=${nagg} sbatch nyu_cassio_aw.sbatch
            done
        done
    done
}

function multirc() {
    export task="multirc"
    export nagg=8
    #export lr=.00003
    export seed=1
    #for seed in 1 2 3; do
    for lr in .00002 .00001 .000005; do
        job_name=${task}-lr${lr}-nagg${nagg}-seed${seed} task=${task} seed=${seed} lr=${lr} nagg=${nagg} sbatch nyu_cassio_aw.sbatch
    done
}

function record() {
    export task="record"
    export lr=.00003
    export nagg=8
    for seed in 1 2 3; do
        job_name=${task}-lr${lr}-nagg${nagg}-seed${seed} task=${task} seed=${seed} lr=${lr} nagg=${nagg} sbatch nyu_cassio_aw.sbatch
    done
}

function rte() {
    export task="rte"
    for lr in .00003; do
        for nagg in 8; do
            for seed in 1 2 3 4 5; do
                job_name=${task}-lr${lr}-nagg${nagg}-seed${seed} task=${task} seed=${seed} lr=${lr} nagg=${nagg} sbatch nyu_cassio_aw.sbatch
            done
        done
    done
}
function wic() {
    export task="wic"
    for lr in .00003; do
        for nagg in 8; do
            for seed in 1 2 3 4 5; do
                job_name=${task}-lr${lr}-nagg${nagg}-seed${seed} task=${task} seed=${seed} lr=${lr} nagg=${nagg} sbatch nyu_cassio_aw.sbatch
            done
        done
    done
}
function wsc() {
    export task="wsc"
    for lr in .00001 .00002 .00003; do
        for nagg in 1 2 4; do
            for seed in 1 2 3 4 5; do
                job_name=${task}-lr${lr}-nagg${nagg}-seed${seed} task=${task} seed=${seed} lr=${lr} nagg=${nagg} sbatch nyu_cassio_aw.sbatch
            done
        done
    done
}

function debug() {
    export task="wsc"
    for lr in .00001; do
        for nagg in 1; do
            for seed in 1; do
                job_name=${task}-lr${lr}-nagg${nagg}-seed${seed} task=${task} seed=${seed} lr=${lr} nagg=${nagg} sbatch nyu_cassio_aw.sbatch
            done
        done
    done
}

echo "MAKE SURE TO ADJUST JOB LENGTH AND MACHINE!!!"
sleep 2
if [ $1 == "debug" ]; then
    debug
elif [ $1 == "boolq" ]; then
    boolq
elif [ $1 == "cb" ]; then
    cb
elif [ $1 == "copa" ]; then
    copa
elif [ $1 == "multirc" ]; then
    multirc
elif [ $1 == "record" ]; then
    record
elif [ $1 == "rte" ]; then
    rte
elif [ $1 == "wic" ]; then
    wic
elif [ $1 == "wsc" ]; then
    wsc
else
    echo "Unknown command ${1}"
fi
