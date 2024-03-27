FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
USER root
RUN apt-get update && apt-get install -y libsndfile1-dev espeak-ng time git
RUN apt-get install -y g++
ENV VIRTUAL_ENV=/usr/local
RUN pip --no-cache-dir install uv
RUN uv venv
RUN uv pip install --no-cache-dir -U pip setuptools
RUN uv pip install --no-cache "pytest<8.0.1" "fsspec>=2023.5.0,<2023.10.0" pytest-subtests pytest-xdist
# END COMMON LAYERS

RUN uv pip install --no-cache-dir --upgrade 'torch<2.2.0' --index-url https://download.pytorch.org/whl/cpu
RUN apt-get install -y  tesseract-ocr
RUN uv pip install --no-cache-dir -U pytesseract python-Levenshtein opencv-python nltk
RUN uv pip install --no-cache-dir natten==0.15.1+torch210cpu -f https://shi-labs.com/natten/wheels
RUN uv pip install --no-cache-dir 'torchvision<0.17' 'torchaudio<2.2.0'
RUN uv pip install  --no-cache-dir "transformers[testing, vision,timm]"  'pip>=21.0.0' 'setuptools>=49.6.0' 'pip[tests]' 'scikit-learn' 'torch-stft' 'nose' 'accelerate' 'dataset'
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN python3 -m pip install --no-cache-dir -e detectron2
RUN pip uninstall -y transformers

RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get autoremove  --purge -y g++
RUN pip cache remove "nvidia-*" 
RUN pip cache remove  triton