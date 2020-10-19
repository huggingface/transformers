#!/usr/bin/env python

import fire

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer


def save_randomly_initialized_version(config_name: str, save_dir: str, **config_kwargs):
    """Save a randomly initialized version of a model using a pretrained config.
    Args:
        config_name: which config to use
        save_dir: where to save the resulting model and tokenizer
        config_kwargs: Passed to AutoConfig

    Usage::
        save_randomly_initialized_version("facebook/bart-large-cnn", "distilbart_random_cnn_6_3", encoder_layers=6, decoder_layers=3, num_beams=3)
    """
    cfg = AutoConfig.from_pretrained(config_name, **config_kwargs)
    model = AutoModelForSeq2SeqLM.from_config(cfg)
    model.save_pretrained(save_dir)
    AutoTokenizer.from_pretrained(config_name).save_pretrained(save_dir)
    return model


if __name__ == "__main__":
    fire.Fire(save_randomly_initialized_version)
