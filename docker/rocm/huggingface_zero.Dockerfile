# Select base Image
FROM rocm/deepspeed:rocm4.1_ubuntu18.04_py3.6_pytorch

# Install dependencies
RUN apt update && apt install -y \
    unzip 
RUN pip3 install regex sacremoses filelock gitpython rouge_score sacrebleu datasets fairscale

# copy repo to workspace
WORKDIR /workspace
COPY . transformers/
RUN cd transformers/ && \
    python3 -m pip install --no-cache-dir .

# set work dir
WORKDIR /workspace/transformers








