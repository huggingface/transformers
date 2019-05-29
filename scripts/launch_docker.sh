#!/bin/bash

nvidia-docker run -ti --privileged --rm -v /home/hdvries/:/home/hdvries -v /home/nathan/:/home/nathan -v /dev/infiniband:/dev/infiniband \
    -w $(pwd) --ipc=host --network=host --dns 1.1.1.1 \
    images.borgy.elementai.lan/hdvries/bert_pytorch $1
