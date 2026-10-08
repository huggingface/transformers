# Builds the CPU CI images, all on top of `ci-base.dockerfile`.
#
# Run from the repository root, e.g.:
#   PYTHON_VERSION=$(cat .python-version) docker buildx bake -f docker/docker-bake.hcl torch-light

# The repo's `.python-version` is the single source for it (bake can't read files).
variable "PYTHON_VERSION" {
  default = ""
  validation {
    condition     = PYTHON_VERSION != ""
    error_message = "Set PYTHON_VERSION, e.g. PYTHON_VERSION=$(cat .python-version)."
  }
}

variable "REF" {
  default = "main"
}

# Appended to the image names, e.g. ":dev".
variable "TAG_SUFFIX" {
  default = ""
}

target "ci-base" {
  context    = "docker"
  dockerfile = "ci-base.dockerfile"
  args       = {
    PYTHON_VERSION = PYTHON_VERSION
  }
}

target "ci-image" {
  name       = image
  matrix     = {
    image = ["quality", "consistency", "custom-tokenizers", "torch-light", "exotic-models", "examples-torch", "pipeline-torch"]
  }
  context    = "docker"
  dockerfile = "${image}.dockerfile"
  contexts   = {
    ci-base = "target:ci-base"
  }
  args       = {
    REF = REF
  }
  tags       = ["huggingface/transformers-${image}${TAG_SUFFIX}"]
}
