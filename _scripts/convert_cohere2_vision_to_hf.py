import argparse
import gc
import logging

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoImageProcessor
from safetensors import safe_open
from pathlib import Path


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def check_all_biases_zero(safetensor_dir, unexpected_keys):
    """
    assert that all unexpected_keys are biases and set to zero
    
    i.e. we are about to remove these and this must be a no-op
    """
    for key in unexpected_keys:
        assert key.endswith("bias"), f"Key {key} is not a bias key."
    logging.info("All unexpected keys are biases")
    keys = set(unexpected_keys)
    logging.info(f"Checking that all unexpected biases are zero in {safetensor_dir}.")
    for safetensor_fp in Path(safetensor_dir).glob("*.safetensors"):
        with safe_open(safetensor_fp, framework='pytorch') as f:
                for k in f.keys():
                    if k in keys:
                        bias = f.get_tensor(k)
                        assert torch.all(bias == 0), f"Bias {k} is not all zeros in {safetensor_fp}. {bias=}"
                        keys.remove(k)
    # assert not keys, f"Not all unexpected biases were found in {safetensor_dir}: {keys}"

def load_and_save_processor(model_path: str, checkpoint_path: str):
    
    logging.info(f"Loading the processor from {checkpoint_path}.")
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    processor.save_pretrained(model_path, safe_serialization=True)
    img_processor = processor.image_processor
    img_processor.save_pretrained(model_path, safe_serialization=True)
    logging.info("Processor saved successfully.")

    # check that it is possible to load the processor from the model path
    loaded_processor = AutoProcessor.from_pretrained(model_path)
    assert isinstance(loaded_processor, type(processor)), "Loaded processor is not of the same type as the original."
    assert isinstance(loaded_processor.image_processor, type(img_processor)), "Loaded image processor is not of the same type as the original."
    logging.info("Processor loaded successfully from the saved path.")

    loaded_img_processor = AutoImageProcessor.from_pretrained(model_path)
    assert isinstance(loaded_img_processor, type(img_processor)), "Loaded image processor is not of the same type as the original."
    logging.info("Image processor loaded successfully from the saved path.")

    # TODO: check tokenizer
    # TODO: hardcode temperature

def write_model(
    model_path: str,
    checkpoint_path: str,
    dtype: torch.dtype = torch.float16,
) -> None:
    """
    Write a model to the specified path.

    Args:
        model_path (str): The path to the model.
        checkpoint_path (str): The path to the checkpoint.
        dtype (torch.dtype): The dtype of the model.
    """
    load_and_save_processor(model_path, checkpoint_path)
    logging.info(f"Loading the model checkpoint from {checkpoint_path}.")

    model, loading_info = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, device_map="auto", output_loading_info=True
    )
    missing, unexpected = loading_info["missing_keys"], loading_info["unexpected_keys"]
    
    # the unexpected keys should all be biases that are set to zero
    if unexpected:
        logging.info("Removing unexpected bias params.")
        check_all_biases_zero(checkpoint_path, unexpected)
        logging.info("Confirmed: all unexpected biases are zero, now re-save checkpoint without them.")
        
    del model.config._name_or_path
    model.config.torch_dtype = dtype
    model = model.to(dtype=dtype)
    logging.info("Saving model in the Transformers format.")
    model.save_pretrained(model_path, safe_serialization=True)
    logging.info("No unexpected keys found, proceeding with the model.")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # check loading works
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", output_loading_info=True
    )
    missing, unexpected = loading_info["missing_keys"], loading_info["unexpected_keys"]

    assert not unexpected, f"Unexpected keys found in the model: {unexpected}"
    assert not missing, f"Missing keys in the model: {missing}"

    logging.info("Model saved and re-loaded successfully.")



def main():
    setup_logging()
    logging.info("Starting the conversion script.")

    parser = argparse.ArgumentParser(description="Convert model checkpoints to Hugging Face format.")
    parser.add_argument("--model_path", type=str, required=True, help="The path to save the converted model.")
    parser.add_argument("--checkpoint_path", required=True, type=str, help="The path to the input tif_export ./poseidon checkpoint.")
    args = parser.parse_args()

    logging.info(f"Converting model from {args.checkpoint_path} to {args.model_path}.")
    write_model(args.model_path, args.checkpoint_path, dtype=torch.float16)
    logging.info("Conversion script completed.")


if __name__ == "__main__":
    # To download the weights from GCP: gsutil -m cp -r gs://cohere-command/experimental_models/c3-sweep-6eoog65n-e0ry-fp16/tif_export/poseidon .
    main()