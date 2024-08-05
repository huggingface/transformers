FROM google/cloud-sdk:slim

# Build args.
ARG GITHUB_REF=refs/heads/main

# TODO: This Dockerfile installs pytorch/xla 3.6 wheels. There are also 3.7
# wheels available; see below.
ENV PYTHON_VERSION=3.6

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates

# Install conda and python.
# NOTE new Conda does not forward the exit status... https://github.com/conda/conda/issues/8385
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b && \
    rm ~/miniconda.sh

ENV PATH=/root/miniconda3/bin:$PATH

RUN conda create -y --name container python=$PYTHON_VERSION

# Run the rest of commands within the new conda env.
# Use absolute path to appease Codefactor.
SHELL ["/root/miniconda3/bin/conda", "run", "-n", "container", "/bin/bash", "-c"]
RUN conda install -y python=$PYTHON_VERSION mkl

RUN pip uninstall -y torch && \
    # Python 3.7 wheels are available. Replace cp36-cp36m with cp37-cp37m
    gsutil cp 'gs://tpu-pytorch/wheels/torch-nightly-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}m-linux_x86_64.whl' . && \
    gsutil cp 'gs://tpu-pytorch/wheels/torch_xla-nightly-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}m-linux_x86_64.whl' . && \
    gsutil cp 'gs://tpu-pytorch/wheels/torchvision-nightly-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}m-linux_x86_64.whl' . && \
    pip install 'torch-nightly-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}m-linux_x86_64.whl' && \
    pip install 'torch_xla-nightly-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}m-linux_x86_64.whl' && \
    pip install 'torchvision-nightly-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}m-linux_x86_64.whl' && \
    rm 'torch-nightly-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}m-linux_x86_64.whl' && \
    rm 'torch_xla-nightly-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}m-linux_x86_64.whl' && \
    rm 'torchvision-nightly-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}m-linux_x86_64.whl' && \
    apt-get install -y libomp5

ENV LD_LIBRARY_PATH=root/miniconda3/envs/container/lib


# Install huggingface/transformers at the current PR, plus dependencies.
RUN git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    git fetch origin $GITHUB_REF:CI && \
    git checkout CI && \
    cd .. && \
    pip install ./transformers && \
    pip install -r ./transformers/examples/pytorch/_test_requirements.txt && \
    pip install pytest

RUN python -c "import torch_xla; print(torch_xla.__version__)"
RUN python -c "import transformers as trf; print(trf.__version__)"
RUN conda init bash
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
