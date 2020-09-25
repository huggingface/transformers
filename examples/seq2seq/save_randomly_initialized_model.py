#!/usr/bin/env python

import fire

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils.logging import get_logger


logger = get_logger(__name__)


def save_randomly_initialized_version(config_name: str, save_dir: str, **config_kwargs):
    """Save a randomly initialized version of a model using a pretrained config.
    Usage:
        save_randomly_initialized_version("facebook/bart-large-cnn", "distilbart_random_cnn_6_3", encoder_layers=6, decoder_layers=3, num_beams=3)
    """
    cfg = AutoConfig.from_pretrained(config_name, **config_kwargs)
    model = AutoModelForSeq2SeqLM.from_config(cfg)
    model.save_pretrained(save_dir)
    AutoTokenizer.from_pretrained(config_name).save_pretrained(save_dir)


if __name__ == "__main__":
    fire.Fire(save_randomly_initialized_version)
