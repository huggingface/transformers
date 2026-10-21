# check=skip=InvalidDefaultArgInFrom
# Common base for the CPU CI images (`*.dockerfile` in this folder).
# Build the images with `docker buildx bake` (see `docker-bake.hcl`), which feeds this
# stage to the other dockerfiles as the `ci-base` build context.
# `PYTHON_VERSION` comes from the repo's `.python-version`.
ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION}-slim
ENV PYTHONDONTWRITEBYTECODE=1
USER root
ENV UV_PYTHON=/usr/local/bin/python
RUN pip --no-cache-dir install uv && uv pip install --no-cache-dir -U pip setuptools
