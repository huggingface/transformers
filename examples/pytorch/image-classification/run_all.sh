#!/usr/bin/env bash

input_file=$1
[[ ! -f $input_file ]] && echo "${input_file} not exist" && exit 1

mkdir -p ./logs

if [[ $HOSTNAME =~ "haca100" ]]; then
    ./mem-daemon-a100 2>&1 >> ./logs/gpu-mem.log &
else
    ./mem-daemon-hac 2>&1 >> ./logs/gpu-mem.log &
fi
daemon_pid=$!

while read model batch_size ; do
    echo Running: $model
    ./run.sh $model $batch_size >/dev/null 2>&1
done < $input_file

kill -9 $daemon_pid
echo Done
