import json
import logging
import os
import subprocess
from argparse import ArgumentParser


logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg.split("=")[0])

    return parser.parse_args()


def main():
    args = parse_args()
    port = 8888
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    num_nodes = len(hosts)
    current_host = os.environ["SM_CURRENT_HOST"]
    rank = hosts.index(current_host)
    os.environ["NCCL_DEBUG"] = "INFO"

    if num_nodes > 1:
        cmd = f"""python -m torch.distributed.launch \
                --nnodes={num_nodes}  \
                --node_rank={rank}  \
                --nproc_per_node={num_gpus}  \
                --master_addr={hosts[0]}  \
                --master_port={port} \
                ./run_glue.py \
                {"".join([f" --{parameter} {value}" for parameter,value in args.__dict__.items()])}"""
    else:
        cmd = f"""python -m torch.distributed.launch \
            --nproc_per_node={num_gpus}  \
            ./run_glue.py \
            {"".join([f" --{parameter} {value}" for parameter,value in args.__dict__.items()])}"""
    try:
        subprocess.run(cmd, shell=True)
    except Exception as e:
        logger.info(e)


if __name__ == "__main__":
    main()
