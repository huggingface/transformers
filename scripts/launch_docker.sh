#!/bin/bash
nvidia-docker run -ti --privileged --rm -v /home/hdvries/:/home/hdvries -v /home/nathan/:/home/nathan -v /dev/infiniband:/dev/infiniband \
    --ipc=host --network=host --dns 1.1.1.1 \
    -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=ALL \
    images.borgy.elementai.lan/hdvries/bert_pytorch bash