import fire
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def remove_prefix(text: str, prefix: str):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text  # or whatever


def convert_pl_to_hf(pl_ckpt_path: str, hf_src_model_dir: str, save_path: str) -> None:
    """Cleanup a pytorch-lightning .ckpt file and save a huggingface model with that state dict. Allows extra pl keys.

    Args:
        pl_ckpt_path: (str) path to a .ckpt file saved by pytorch_lightning, e.g.
        hf_src_model_dir: (str) path to a directory containing a correctly shaped checkpoint
        save_path: (str) directory to save the new model

    """
    hf_model = AutoModelForSeq2SeqLM.from_pretrained(hf_src_model_dir)
    state_dict = {
        remove_prefix(k, "model."): v for k, v in torch.load(pl_ckpt_path, map_location="cpu")["state_dict"].items()
    }

    missing, unexpected = hf_model.load_state_dict(state_dict, strict=False)
    assert not missing, f"missing keys: {missing}"
    hf_model.save_pretrained(save_path)
    try:
        tok = AutoTokenizer.from_pretrained(hf_src_model_dir)
        tok.save_pretrained(save_path)
    except Exception:
        pass
        # dont copy tokenizer if cant


if __name__ == "__main__":
    fire.Fire(convert_pl_to_hf)
