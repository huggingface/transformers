FROM ci-base
ARG REF=main
RUN apt-get update && apt-get install -y time git make
RUN uv pip install --no-cache-dir -U pip setuptools GitPython "git+https://github.com/huggingface/transformers.git@${REF}#egg=transformers[quality]" urllib3
RUN apt-get install -y jq curl && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN uv pip install --no-cache-dir 'transformers-ci[otel] @ git+https://github.com/huggingface/transformers-ci@main'
