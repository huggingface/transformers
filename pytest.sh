#!/bin/bash

export PYTHONDONTWRITEBYTECODE=1 && export PYTHONUNBUFFERED=1 && export OMP_NUM_THREADS=1 && export TRANSFORMERS_IS_CI=true && export PYTEST_TIMEOUT=120 && export RUN_PIPELINE_TESTS=false && export RUN_FLAKY=true
python3 -m pytest -m 'not generate' -n 2 --max-worker-restart=0 -rsfE tests/models/imagegpt/test_modeling_imagegpt.py tests/models/maskformer/test_modeling_maskformer.py