#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import shlex

import runhouse as rh


if __name__ == "__main__":
    # Refer to https://runhouse-docs.readthedocs-hosted.com/en/main/rh_primitives/cluster.html#hardware-setup for cloud access
    # setup instructions, if using on-demand hardware

    # If user passes --user <user> --host <host> --key_path <key_path> <example> <args>, fill them in as BYO cluster
    # If user passes --instance <instance> --provider <provider> <example> <args>, fill them in as on-demand cluster
    # Throw an error if user passes both BYO and on-demand cluster args
    # Otherwise, use default values
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, default="ubuntu")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--key_path", type=str, default=None)
    parser.add_argument("--instance", type=str, default="V100:1")
    parser.add_argument("--provider", type=str, default="cheapest")
    parser.add_argument("--use_spot", type=bool, default=False)
    parser.add_argument("--example", type=str, default="pytorch/text-generation/run_generation.py")
    args, unknown = parser.parse_known_args()
    if args.host != "localhost":
        if args.instance != "V100:1" or args.provider != "cheapest":
            raise ValueError("Cannot specify both BYO and on-demand cluster args")
        cluster = rh.cluster(
            name="rh-cluster", ips=[args.host], ssh_creds={"ssh_user": args.user, "ssh_private_key": args.key_path}
        )
    else:
        cluster = rh.cluster(
            name="rh-cluster", instance_type=args.instance, provider=args.provider, use_spot=args.use_spot
        )
    example_dir = args.example.rsplit("/", 1)[0]

    # Set up remote environment
    cluster.install_packages(["pip:./"])  # Installs transformers from local source
    # Note transformers is copied into the home directory on the remote machine, so we can install from there
    cluster.run([f"pip install -r transformers/examples/{example_dir}/requirements.txt"])
    cluster.run(["pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu117"])

    # Run example. You can bypass the CLI wrapper and paste your own code here.
    cluster.run([f'python transformers/examples/{args.example} {" ".join(shlex.quote(arg) for arg in unknown)}'])

    # Alternatively, we can just import and run a training function (especially if there's no wrapper CLI):
    # from my_script... import train
    # reqs = ['pip:./', 'torch', 'datasets', 'accelerate', 'evaluate', 'tqdm', 'scipy', 'scikit-learn', 'tensorboard']
    # launch_train_gpu = rh.function(fn=train,
    #                                system=gpu,
    #                                reqs=reqs,
    #                                name='train_bert_glue')
    #
    # We can pass in arguments just like we would to a function:
    # launch_train_gpu(num_epochs = 3, lr = 2e-5, seed = 42, batch_size = 16
    #                  stream_logs=True)
