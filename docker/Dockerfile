FROM nvcr.io/nvidia/pytorch:19.04-py3

RUN pip install boto3 sentencepiece

RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext

WORKDIR /src

CMD ["/bin/bash"]
