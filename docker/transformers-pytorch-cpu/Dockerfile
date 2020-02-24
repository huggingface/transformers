FROM continuumio/miniconda3:4.7.12-alpine

ARG PYTHON_VERSION=3.7
USER root
RUN apk add --no-cache --virtual=build-dependencies \
                       git \
                       curl \
                       ca-certificates

RUN /opt/conda/bin/conda update -n base -c defaults conda && \
     /opt/conda/bin/conda create -n pytorch python==$PYTHON_VERSION && \
     /opt/conda/bin/conda install -y -n pytorch -c pytorch pytorch cpuonly && \
     /opt/conda/bin/conda install -y -n pytorch jupyter && \
     /opt/conda/bin/conda clean -ya && \
     apk del build-dependencies && \
     echo 'export PATH=/opt/conda/bin:$PATH' >> /etc/profile.d/conda.sh
ENV PATH /opt/conda/envs/pytorch/bin:/opt/conda/bin:$PATH


WORKDIR /workspace
RUN chmod -R a+w .

ENTRYPOINT ["jupyter", "notebook", "--allow-root", "--ip", "0.0.0.0"]