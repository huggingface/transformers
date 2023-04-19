#!/usr/bin/env bash


if [[ $HOSTNAME =~ "haca100" ]]; then
    ./mem-daemon-a100 2>&1 >> gpu-mem.log &
else
    ./mem-daemon-hac 2>&1 >> gpu-mem.log &
fi
daemon_pid=$!

for model in vit deit swin swinv2; do
    echo Running $model
    ./mim.sh $model > /dev/null 2>&1
done

echo "Running MAE"
./mae.sh > /dev/null 2>&1

kill -9 $daemon_pid
echo Done
