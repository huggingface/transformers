import os
import requests
import transformers
from datetime import date, timedelta
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN", None)

__all__ = ["HF_TOKEN", "clean_model_id", "fetch_latest_ci_results", "get_config_and_class"]

CI_DATASET = "hf-internal-testing/transformers_daily_ci"
CI_PATH = "ci_results_run_models_gpu/model_results.json"


def clean_model_id(raw: str) -> str:
    raw = raw.strip()
    if "huggingface.co/" in raw:
        raw = raw.split("huggingface.co/")[-1].strip("/")
    return raw


def fetch_latest_ci_results() -> tuple:
    today = date.today()
    for delta in range(3):
        d = (today - timedelta(days=delta)).strftime("%Y-%m-%d")
        url = f"https://huggingface.co/datasets/{CI_DATASET}/resolve/main/{d}/{CI_PATH}"
        try:
            resp = requests.get(url, timeout=15, allow_redirects=True)
            if resp.status_code == 200:
                return d, resp.json()
        except Exception:
            continue
    return None, None


def get_config_and_class(model_id):
    """Return (config, model_cls, arch). model_cls is None if the import fails."""
    config = AutoConfig.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
    arch = (config.architectures or ["unknown"])[0]
    # transformers uses lazy imports; getattr can raise ModuleNotFoundError
    # (e.g. missing optional dep) instead of returning the default — catch it.
    try:
        model_cls = getattr(transformers, arch)
    except Exception:
        model_cls = None
    return config, model_cls, arch
