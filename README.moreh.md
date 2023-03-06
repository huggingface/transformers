# [Moreh] Running HuggingFace `transformers` in Moreh AI Framework


## Prepare

### Code
```bash
git clone https://github.com/loctxmoreh/transformers
```

### Environment

The `transformers` library is installed in *editable* mode so changes can be
made directly into the source in order to take effect.

#### On A100 machine
```bash
conda env create -f a100env.yml
conda activate transformers-dev
```

#### On HAC machine
```bash
conda env create -f hacenv.yml
conda activate transformers-dev
update-moreh --force --tensorflow
```
