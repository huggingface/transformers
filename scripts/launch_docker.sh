#!/bin/bash

nvidia-docker run -ti --privileged --rm -v /home/hdvries/:/home/hdvries -v /home/nathan/:/home/nathan -v /dev/infiniband:/dev/infiniband \
    -w $(pwd) --ipc=host --network=host -e NCCL_DEBUG=INFO -e NCCL_SOCKET_IFNAME="^br,lo" --dns 192.168.170.100 \
    images.borgy.elementai.lan/hdvries/bert_pytorch $1
