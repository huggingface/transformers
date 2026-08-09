FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

# Use login shell to read variables from `~/.profile` (to pass dynamic created variables between RUN commands)
SHELL ["sh", "-lc"]

# The following `ARG` are mainly used to specify the versions explicitly & directly in this docker file, and not meant
# to be used as arguments for docker build (so far).

# Keep this in sync with `transformers-all-latest-gpu` (the model CI image). It must also satisfy the
# quantization backends installed below -- notably `compressed-tensors`, which requires `torch>=2.10`.
# Pinning an older torch (e.g. 2.8.0) makes `compressed-tensors` uninstall the pinned torch mid-build and
# pull a newer one, churning the torch stack across RUN layers and leaving `torchaudio` ABI-mismatched
# (see the Jul 2026 daily quantization CI collapse). The smoke test at the end of this file guards it.
ARG PYTORCH='2.11.0'
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu126'

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3 python3-pip ffmpeg
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; echo "export VERSION='$VERSION'" >> ~/.profile
RUN echo torch=$VERSION
# `torchvision` and `torchaudio` should be installed along with `torch`, especially for nightly build.
# `torchaudio` is pinned to the same version as `torch`: torchaudio ships a compiled C extension
# (`_torchaudio.abi3.so`) built against one exact torch ABI, so a floating/mismatched torchaudio fails to
# load with `OSError: ... _torchaudio.abi3.so: undefined symbol`. (transformers now imports torchaudio
# lazily -- see loss/loss_rnnt.py -- so a mismatch no longer breaks `import transformers`, but it would
# still leave torchaudio unusable and broke the daily quantization CI in Jul 2026. Keep them matched.)
# Mirrors the `torchaudio==${PYTORCH}.*` pin in transformers-all-latest-gpu.
#
# TODO(torchaudio-cap): torchaudio 2.11 is the LAST release -- pytorch/audio has stopped publishing (I/O
# moved to TorchCodec, see https://github.com/pytorch/audio/issues/3902). 2.11 is marked compatible with
# future torch, but there is no torchaudio > 2.11, so `torchaudio==${PYTORCH}.*` resolves only while
# PYTORCH <= 2.11; the next bump past 2.11 fails here with "no matching distribution". At that point this
# pin (and the torchaudio-backed RNN-T path in loss/loss_rnnt.py) needs a new plan -- cap torchaudio at
# 2.11, or drop the torchaudio dependency.
RUN [ ${#PYTORCH} -gt 0 ] && TORCHAUDIO='torchaudio=='$PYTORCH'.*' || TORCHAUDIO='torchaudio'; \
    python3 -m pip install --no-cache-dir -U $VERSION torchvision $TORCHAUDIO --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/accelerate@main#egg=accelerate

# Add optimum for gptq quantization testing
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/optimum@main#egg=optimum

# Add PEFT
RUN python3 -m pip install --no-cache-dir git+https://github.com/huggingface/peft@main#egg=peft

# needed in bnb and awq
RUN python3 -m pip install --no-cache-dir einops

# Add bitsandbytes
RUN python3 -m pip install --no-cache-dir bitsandbytes

# # Add gptqmodel
# RUN python3 -m pip install --no-cache-dir gptqmodel

# Add hqq for quantization testing
RUN python3 -m pip install --no-cache-dir hqq

# For GGUF tests
RUN python3 -m pip install --no-cache-dir gguf

# Add quanto for quantization testing
RUN python3 -m pip install --no-cache-dir optimum-quanto

# Add compressed-tensors for quantization testing
RUN python3 -m pip install --no-cache-dir compressed-tensors

# Add AMD Quark for quantization testing
RUN python3 -m pip install --no-cache-dir amd-quark

# Add AutoRound for quantization testing
RUN python3 -m pip install --no-cache-dir auto-round

# Add torchao for quantization testing
RUN python3 -m pip install --no-cache-dir torchao

# Add transformers in editable mode
RUN python3 -m pip install --no-cache-dir -e ./transformers[dev]

# `kernels` may give different outputs (within 1e-5 range) even with the same model (weights) and the same inputs
RUN python3 -m pip uninstall -y kernels

# Uninstall flash-attn installed by autoawq, it causes issues here : https://github.com/huggingface/transformers/actions/runs/15915442841/job/44892146131
RUN python3 -m pip uninstall -y flash-attn

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

# Add fouroversix for quantization testing.
# fouroversix ships ONLY source distributions (no wheels, any version), and its FP4 GEMM/quant CUDA
# kernels are Blackwell-only: they instantiate `cutlass::float_e2m1_t` with SM100/SM120 TMA descriptors
# and require CUDA >= 12.8. This image is CUDA 12.6 and the quant CI runs on aws-g5 (A10G, sm_86,
# Ampere), so a from-source build cannot compile the kernels -- nvcc aborts with
# `error: static assertion failed with "Unknown TMA Format!"` for any pre-Hopper arch -- and even a
# successful build could never run FP4 kernels on an A10G.
#
# We therefore install with `SKIP_CUDA_BUILD=1`, which sets `ext_modules = None` (no compilation) and
# ships fouroversix's pure-PyTorch/Triton reference backends, the path it provides for non-Blackwell
# hardware. The package still imports, so `is_fouroversix_available()` stays True and the integration
# tests run against the reference backend; the CUDA/cutlass backend simply reports itself unavailable
# (`matmul/cutlass/backend.py` guards `import fouroversix._C` behind a Blackwell + try/except check).
# `--no-build-isolation` is required because fouroversix's `setup.py` imports `torch` at build time.
# When a Blackwell (sm100/sm120) quant CI image on CUDA >= 12.8 exists, drop SKIP_CUDA_BUILD there so
# the real FP4 kernels are exercised.
RUN python3 -m pip install --no-cache-dir --upgrade "setuptools>=77.0.3" wheel
RUN SKIP_CUDA_BUILD=1 python3 -m pip install --no-cache-dir "fouroversix>=1.0.2" --no-build-isolation

# Add fp-quant for quantization testing
RUN python3 -m pip install --no-cache-dir "fp-quant>=0.3.2"

# Smoke test: fail the image build immediately if the pinned torch stack was clobbered by a later
# `pip install`, or if a compiled extension can't load against the pinned torch ABI. This runs after
# every install above, so a dep that quietly drags in a mismatched torch/torchaudio is caught here.
# `import transformers` walks the eager `modeling_utils -> loss_utils -> loss_rnnt -> torchaudio`
# import chain, so a torchaudio ABI mismatch (which otherwise only surfaces as broken pytest
# collection for every quantization job, see the Jul 2026 daily CI collapse) fails the build instead
# of shipping a silently-broken image. No GPU is present at build time, so we only exercise imports
# and the CUDA runtime load, not device availability.
RUN set -e; \
    python3 -c "import torch; print('torch version:', torch.__version__); torch.cuda.is_available()"; \
    python3 -c "import torch, sys; v = torch.__version__.split('+')[0]; sys.exit(0 if v.startswith('${PYTORCH}') else 'ERROR: torch is ' + torch.__version__ + ', expected ${PYTORCH}.* - the pinned CUDA build was clobbered')"; \
    python3 -c "import torch, torchaudio; print('torchaudio version:', torchaudio.__version__)"; \
    python3 -c "import transformers; print('transformers imports OK:', transformers.__version__)"

# Low usage or incompatible lib, will enable later on

# # Add aqlm for quantization testing
# RUN python3 -m pip install --no-cache-dir aqlm[gpu]==1.0.2

# # Add vptq for quantization testing
# RUN pip install vptq

# Add spqr for quantization testing
# Commented for now as No matching distribution found we need to reach out to the authors
# RUN python3 -m pip install --no-cache-dir spqr_quant[gpu]

# # Add eetq for quantization testing
# RUN git clone https://github.com/NetEase-FuXi/EETQ.git && cd EETQ/ && git submodule update --init --recursive && pip install .

# # Add flute-kernel and fast_hadamard_transform for quantization testing
# # Commented for now as they cause issues with the build
# # TODO: create a new workflow to test them
# RUN python3 -m pip install --no-cache-dir flute-kernel==0.4.1
# RUN python3 -m pip install --no-cache-dir git+https://github.com/Dao-AILab/fast-hadamard-transform.git
