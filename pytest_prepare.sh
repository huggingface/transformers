#!/bin/bash

export PYTHONDONTWRITEBYTECODE=1 && export PYTHONUNBUFFERED=1 && export OMP_NUM_THREADS=1 && export TRANSFORMERS_IS_CI=true && export PYTEST_TIMEOUT=120 && export RUN_PIPELINE_TESTS=false && export RUN_FLAKY=true && UV_PYTHON=/usr/bin/python3.9-dbg

git clone https://github.com/huggingface/transformers.git
cd transformers/
git checkout debug_too_long_no_output


python3 -V
python3 -c "import sys; print(sys.executable)"


apt-get update
apt-get install -y gdb
apt-get remove --purge needrestart -y
echo 'deb http://deb.debian.org/debian bullseye main' >> /etc/apt/sources.list
apt-get update
apt-get install -y python3.9-dbg
sed -i '/bullseye/d' /etc/apt/sources.list


which python3
which python3.9-dbg
ls -la /usr/bin/python*


python3 -V
python3 -c "import sys; print(sys.executable)"


alias python3='python3.9-dbg'


which python3
python3 -V
python3 -c "import sys; print(sys.executable)"

echo $UV_PYTHON


python3 -m pip --no-cache-dir install uv && uv pip install --no-cache-dir -U pip setuptools
uv pip install --no-cache-dir 'torch' 'torchaudio' 'torchvision' 'torchcodec' --index-url https://download.pytorch.org/whl/cpu
uv pip install --no-deps timm accelerate --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install --no-cache-dir librosa "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[sklearn,sentencepiece,vision,testing,tiktoken,num2words,video]"
uv pip install git+https://github.com/ydshieh/pytest.git@8.3.5-ydshieh

uv pip install .

mkdir test-results
python3 utils/fetch_hub_objects_for_ci.py
