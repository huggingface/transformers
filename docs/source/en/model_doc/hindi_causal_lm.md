# HindiCausalLM

## Overview

The HindiCausalLM model is a Hindi language model developed by [ConvAI Innovations](https://huggingface.co/convaiinnovations). It's a causal language model designed specifically for Hindi language understanding and generation.

The model implementation is based on the transformer architecture with configurable activation functions, positional encoding, and normalization layers.

## HindiCausalLMConfig

[[autodoc]] HindiCausalLMConfig

## HindiCausalLMTokenizer

[[autodoc]] HindiCausalLMTokenizer
    - __call__
    - save_vocabulary

## HindiCausalLMModel

[[autodoc]] HindiCausalLMModel
    - forward

## HindiCausalLMHeadModel

[[autodoc]] HindiCausalLMHeadModel
    - forward