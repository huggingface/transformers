FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN pip install transformers jupyter

EXPOSE 5000

ENTRYPOINT jupyter notebook --port 5000