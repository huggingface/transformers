import argparse
import os

import torch
import torch.distributed as dist


# Environment variables set by torch.distributed.launch
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])

LOCAL_RANK = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
WORLD_SIZE = int(os.environ["OMPI_COMM_WORLD_SIZE"])
WORLD_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])


def run(backend):
    tensor = torch.zeros(1)
    # Need to put tensor on a GPU device for nccl backend
    if backend == "nccl":
        device = torch.device("cuda:{}".format(LOCAL_RANK))
        tensor = tensor.to(device)

    if WORLD_RANK == 0:
        for rank_recv in range(1, WORLD_SIZE):
            dist.send(tensor=tensor, dst=rank_recv)
            print("worker_{} sent data to Rank {}\n".format(0, rank_recv))
    else:
        dist.recv(tensor=tensor, src=0)
        print("worker_{} has received data from rank {}\n".format(WORLD_RANK, 0))


def init_processes(backend):
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE)
    run(backend)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility."
    )
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"])
    args = parser.parse_args()

    init_processes(backend=args.backend)

""""
python-m torch.distributed.launch \
--nproc_per_node=2 --nnodes=2 --node_rank=0 \
test_compile.py

python3 -m torch.distributed.launch \
--nproc_per_node=2 --nnodes=2 --node_rank=1 \
--master_addr=104.171.200.62 --master_port=1234 \
main.py \
--backend=nccl --use_syn --batch_size=8192 --arch=resnet152



mpirun -np 4 \
-H 104.171.200.62:2,104.171.200.182:2 \
-x MASTER_ADDR=104.171.200.62 \
-x MASTER_PORT=1234 \
-x PATH \
-bind-to none -map-by slot \
-mca pml ob1 -mca btl ^openib \
python3 main.py
"""


""""
You need a host file with the name of hosts.
for example I have arthur@ip-26-0-162-46 and arthur@ip-26-0-162-239

________
hostfile
ip-26-0-162-46 slots=8
ip-26-0-162-239 slots=8
________

mpirun --hostfile hostfile -np 16 \
    --bind-to none --map-by slot \
    -x MASTER_ADDR=<master-node-ip> \
    -x MASTER_PORT=29500 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=^lo,docker0 \
    -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python your_script.py --backend nccl


to get the master IP you need to do a few things:
hostname -I | awk '{print $1}'


Use `ping ip-26-0-162-46` to check if connected

26.0.162.46

mpirun --hostfile hostfile -np 16 \
    --bind-to none --map-by slot \
    -x MASTER_ADDR=26.0.162.46 \
    -x MASTER_PORT=29500 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=^lo,docker0 \
    -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python your_script.py --backend nccl


mpirun --hostfile hostfile -np 2     -x NCCL_DEBUG=INFO     python -c "import os;print(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])" -b 8 -e 128M -f 2 -g 1
to test your setup
"""
