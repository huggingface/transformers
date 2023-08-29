# Select base Image
FROM huggingface/transformers-pytorch-gpu

# Install dependencies
RUN apt update && apt install -y \
    unzip 
RUN pip3 install regex sacremoses filelock gitpython rouge_score sacrebleu datasets fairscale deepspeed

# copy repo to workspace
WORKDIR /workspace
COPY . transformers/
RUN cd transformers/ && \
    python3 -m pip install --no-cache-dir .

# set work dir
WORKDIR /workspace/transformers








