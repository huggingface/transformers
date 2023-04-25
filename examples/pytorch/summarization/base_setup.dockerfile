FROM python:3.10 

RUN apt-get -y update
RUN apt-get -y install git

WORKDIR /app

RUN git clone https://github.com/huggingface/transformers.git

WORKDIR /app/transformers

#point head to last working version
RUN git checkout 1b1867d 

WORKDIR /app/transformers/examples/pytorch/summarization

COPY requirements.txt .
#install requirements
RUN pip install -r requirements.txt
