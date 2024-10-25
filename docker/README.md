# Dockers for `transformers`

In this folder you will find various docker files, and some subfolders. 
- dockerfiles (ex: `consistency.dockerfile`) present under `~/docker` are used for our "fast" CIs. You should be able to use them for tasks that only need CPU. For example `torch-light` is a very light weights container (703MiB). 
- subfloder contain dockerfiles used for our `slow` CIs, which *can* be used for GPU tasks, but they are **BIG** as they were not specifically designed for a single model / single task. Thus the `~/docker/transformers-pytorch-gpu` includes additional dependencies to allow us to run ALL model tests (say `librosa` or `tesseract`, which you do not need to run LLMs)

Note that in both case, you need to run `uv pip install -e .`, which should take around 5 seconds. We do it outside the dockerfile for the need of our CI: we checkout a new branch each time, and the `transformers` code is thus updated. 

We are open to contribution, and invite the community to create dockerfiles with potential arguments that properly choose extras depending on the model's dependencies! :hugs: 