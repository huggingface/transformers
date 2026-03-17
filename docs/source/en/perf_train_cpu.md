<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CPU

CPU training works well when GPUs aren't available or when you want a cost-effective setup. Modern Intel CPUs support bf16 mixed precision training with [PyTorch's AMP](https://docs.pytorch.org/docs/stable/amp) (Automatic Mixed Precision) for CPU backends, which reduces memory usage and speeds up training.

Scale CPU training across multiple sockets or nodes if a single CPU is too slow. The examples below cover three scenarios:

- a single CPU
- multiple processes on one machine (one per CPU socket)
- multiple processes across several machines

All distributed examples use [Intel MPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html) from the [Intel oneAPI HPC Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html) for communication and a DDP strategy with [`Trainer`].

<hfoptions id="distrib-cpu">
<hfoption id="single CPU">

[`Trainer`] supports bf16 mixed precision training on CPU. Prefer bf16 over fp16 for CPU training because it's more numerically stable. Pass `--bf16` to enable PyTorch's CPU autocast and `--use_cpu` to force CPU training. The example below runs the [run_qa.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) script.

```bash
python run_qa.py \
 --model_name_or_path google-bert/bert-base-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12 \
 --learning_rate 3e-5 \
 --num_train_epochs 2 \
 --max_seq_length 384 \
 --doc_stride 128 \
 --output_dir /tmp/debug_squad/ \
 --bf16 \
 --use_cpu
```

You can pass the same parameters to [`TrainingArguments`] directly.

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,
    use_cpu=True,
)
```

</hfoption>
<hfoption id="distributed CPU (single node)">

On a dual-socket CPU, run one process per socket. Keeping memory accesses local to each socket improves throughput. The example below launches two processes on a single machine with one process per socket.

> [!TIP]
> Set `OMP_NUM_THREADS` to the number of physical cores on one socket, minus one core reserved for the OS. For example, on a 24-core socket set `OMP_NUM_THREADS=23`.

```bash
export MASTER_ADDR=127.0.0.1
mpirun -n 2 -genv OMP_NUM_THREADS=23 \
python3 run_qa.py \
 --model_name_or_path google-bert/bert-large-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12 \
 --learning_rate 3e-5 \
 --num_train_epochs 2 \
 --max_seq_length 384 \
 --doc_stride 128 \
 --output_dir /tmp/debug_squad/
```

</hfoption>
<hfoption id="distributed CPU (multiple nodes)">

Scale training to four processes across two Xeon machines (`node0` and `node1`) using a hostfile that lists each node's IP address. The `-n 4` flag sets the total number of processes. `-ppn 2` sets two processes per node (one per socket).

Run this script from `node0`, which acts as the main process.

> [!TIP]
> Set `OMP_NUM_THREADS` to the number of physical cores on one socket, minus one core reserved for the OS. For example, on a 24-core socket set `OMP_NUM_THREADS=23`.

Create a hostfile with the IP address of each node.

```bash
cat hostfile
xxx.xxx.xxx.xxx #node0 ip
xxx.xxx.xxx.xxx #node1 ip
```

Then run the training script.

```bash
export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
mpirun -f hostfile -n 4 -ppn 2 \
 -genv OMP_NUM_THREADS=23 \
python3 run_qa.py \
 --model_name_or_path google-bert/bert-large-uncased \
 --dataset_name squad \
 --do_train \
 --do_eval \
 --per_device_train_batch_size 12 \
 --learning_rate 3e-5 \
 --num_train_epochs 2 \
 --max_seq_length 384 \
 --doc_stride 128 \
 --output_dir /tmp/debug_squad/ \
 --use_cpu \
 --bf16
```

</hfoption>
</hfoptions>

## Kubernetes

Distributed CPU training can also run on a Kubernetes cluster using [PyTorchJob](https://www.kubeflow.org/docs/components/training/user-guides/pytorch/).

Complete the following setup before deploying:

1. Ensure you have access to a Kubernetes cluster with [Kubeflow](https://www.kubeflow.org/docs/started/installing-kubeflow/) installed.
2. Install and configure [kubectl](https://kubernetes.io/docs/tasks/tools) to interact with the cluster.
3. Set up a [PersistentVolumeClaim (PVC)](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) to store datasets and model files.
4. Build a Docker image for the training script and its dependencies.

The example Dockerfile below starts from an Intel-optimized PyTorch base image that includes MPI support for multi-node CPU training. It also installs two performance libraries:

- `google-perftools` (`libtcmalloc`): a memory allocator that reduces allocation overhead compared to the default system allocator.
- `libomp-dev` (`libiomp5`): Intel's OpenMP runtime, which provides better CPU thread management than the GNU OpenMP default.

```dockerfile
FROM intel/intel-optimized-pytorch:2.4.0-pip-multinode

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
    google-perftools \
    libomp-dev

WORKDIR /workspace

# Download and extract the transformers code
ARG HF_TRANSFORMERS_VER="4.46.0"
RUN pip install --no-cache-dir \
    transformers==${HF_TRANSFORMERS_VER} && \
    mkdir transformers && \
    curl -sSL --retry 5 https://github.com/huggingface/transformers/archive/refs/tags/v${HF_TRANSFORMERS_VER}.tar.gz | tar -C transformers --strip-components=1 -xzf -
```

Build the image and push it to a registry accessible from your cluster's nodes before deploying.

### PyTorchJob

[PyTorchJob](https://www.kubeflow.org/docs/components/training/user-guides/pytorch/) is a Kubernetes custom resource that manages PyTorch distributed training jobs. It handles worker pod lifecycle, restart policy, and process coordination. Your training script only needs to handle the model and data.

The example yaml file below sets up four workers running the [run_qa.py](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering) script with bf16 enabled.

When setting CPU resource limits and requests, use [CPU units](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-cpu) where one unit equals one physical core or virtual core. Set both limits and requests to the same value for a `Guaranteed` [quality of service](https://kubernetes.io/docs/tasks/configure-pod-container/quality-service-pod). Leave some cores unallocated for kubelet and system processes. Set `OMP_NUM_THREADS` to match the number of allocated CPU units so PyTorch uses all available cores.

Adapt the yaml file based on your training script and the number of nodes in your cluster.

```yaml
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: transformers-pytorchjob
spec:
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 1
    maxReplicas: 4
    maxRestarts: 10
  pytorchReplicaSpecs:
    Worker:
      replicas: 4  # The number of worker pods
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: <image name>:<tag>  # Specify the docker image to use for the worker pods
              imagePullPolicy: IfNotPresent
              command: ["/bin/bash", "-c"]
              args:
                - >-
                  cd /workspace/transformers;
                  pip install -r /workspace/transformers/examples/pytorch/question-answering/requirements.txt;
                  torchrun /workspace/transformers/examples/pytorch/question-answering/run_qa.py \
                    --model_name_or_path distilbert/distilbert-base-uncased \
                    --dataset_name squad \
                    --do_train \
                    --do_eval \
                    --per_device_train_batch_size 12 \
                    --learning_rate 3e-5 \
                    --num_train_epochs 2 \
                    --max_seq_length 384 \
                    --doc_stride 128 \
                    --output_dir /tmp/pvc-mount/output_$(date +%Y%m%d_%H%M%S) \
                    --bf16;
              env:
              - name: LD_PRELOAD
                value: "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4.5.9:/usr/local/lib/libiomp5.so"
              - name: HF_HUB_CACHE
                value: "/tmp/pvc-mount/hub_cache"
              - name: HF_DATASETS_CACHE
                value: "/tmp/pvc-mount/hf_datasets_cache"
              - name: LOGLEVEL
                value: "INFO"
              - name: OMP_NUM_THREADS  # Set to match the number of allocated CPU units
                value: "240"
              resources:
                limits:
                  cpu: 240  # Update the CPU and memory limit values based on your nodes
                  memory: 128Gi
                requests:
                  cpu: 240  # Update the CPU and memory request values based on your nodes
                  memory: 128Gi
              volumeMounts:
              - name: pvc-volume
                mountPath: /tmp/pvc-mount
              - mountPath: /dev/shm
                name: dshm
          restartPolicy: Never
          nodeSelector:  # Optionally use nodeSelector to match a certain node label for the worker pods
            node-type: gnr
          volumes:
          - name: pvc-volume
            persistentVolumeClaim:
              claimName: transformers-pvc
          - name: dshm
            emptyDir:
              medium: Memory
```

### Deploy

Deploy the PyTorchJob to the cluster with the following command.

```bash
export NAMESPACE=<specify your namespace>

kubectl create -f pytorchjob.yaml -n ${NAMESPACE}
```

List the pods in the namespace to monitor their status. Pods start as **Pending** while the container image is pulled, then transition to **Running**.

```bash
kubectl get pods -n ${NAMESPACE}

NAME                                                     READY   STATUS                  RESTARTS          AGE
...
transformers-pytorchjob-worker-0                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-1                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-2                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-3                         1/1     Running                 0                 7m37s
...
```

Stream logs from a worker pod to follow training progress.

```bash
kubectl logs transformers-pytorchjob-worker-0 -n ${NAMESPACE} -f
```

Copy the trained model from the PVC or your storage location after training completes. Then delete the PyTorchJob resource.

```bash
kubectl delete -f pytorchjob.yaml -n ${NAMESPACE}
```

## Next steps

- Read the [Accelerating PyTorch Transformers with Intel Sapphire Rapids](https://huggingface.co/blog/intel-sapphire-rapids) blog post for a deeper look at BF16 performance on modern Intel hardware.
