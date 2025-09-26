import os
from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="huggingface-cache.tar.gz",
    path_in_repo="huggingface-cache.tar.gz",
    repo_id="hf-internal-testing/hf_hub_cache",
    token=os.environ["TRANSFORMERS_CI_RESULTS_UPLOAD_TOKEN"],
    repo_type="dataset",
)
