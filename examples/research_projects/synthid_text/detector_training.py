# coding=utf-8
# Copyright 2024 Google DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import dataclasses
import enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BayesianDetectorConfig,
    BayesianDetectorModel,
    SynthIDTextWatermarkDetector,
    SynthIDTextWatermarkingConfig,
    SynthIDTextWatermarkLogitsProcessor,
)
from utils import (
    get_tokenized_uwm_outputs,
    get_tokenized_wm_outputs,
    process_raw_model_outputs,
    update_fn_if_fpr_tpr,
    upload_model_to_hf,
)


@enum.unique
class ValidationMetric(enum.Enum):
    """Direction along the z-axis."""

    TPR_AT_FPR = "tpr_at_fpr"
    CROSS_ENTROPY = "cross_entropy"


@dataclasses.dataclass
class TrainingArguments:
    """Training arguments pertaining to the training loop itself."""

    eval_metric: Optional[str] = dataclasses.field(
        default=ValidationMetric.TPR_AT_FPR, metadata={"help": "The evaluation metric used."}
    )


def train_detector(
    detector: torch.nn.Module,
    g_values: torch.Tensor,
    mask: torch.Tensor,
    watermarked: torch.Tensor,
    epochs: int = 250,
    learning_rate: float = 1e-3,
    minibatch_size: int = 64,
    seed: int = 0,
    l2_weight: float = 0.0,
    shuffle: bool = True,
    g_values_val: Optional[torch.Tensor] = None,
    mask_val: Optional[torch.Tensor] = None,
    watermarked_val: Optional[torch.Tensor] = None,
    verbose: bool = False,
    validation_metric: ValidationMetric = ValidationMetric.TPR_AT_FPR,
) -> Tuple[Dict[str, Any], float]:
    """Trains a Bayesian detector model.

    Args:
      g_values: g-values of shape [num_train, seq_len, watermarking_depth].
      mask: A binary array shape [num_train, seq_len] indicating which g-values
        should be used. g-values with mask value 0 are discarded.
      watermarked: A binary array of shape [num_train] indicating whether the
        example is watermarked (0: unwatermarked, 1: watermarked).
      epochs: Number of epochs to train for.
      learning_rate: Learning rate for optimizer.
      minibatch_size: Minibatch size for training. Note that a minibatch
        requires ~ 32 * minibatch_size * seq_len * watermarked_depth *
        watermarked_depth bits of memory.
      seed: Seed for parameter initialization.
      l2_weight: Weight to apply to L2 regularization for delta parameters.
      shuffle: Whether to shuffle before training.
      g_values_val: Validation g-values of shape [num_val, seq_len,
        watermarking_depth].
      mask_val: Validation mask of shape [num_val, seq_len].
      watermarked_val: Validation watermark labels of shape [num_val].
      verbose: Boolean indicating verbosity of training. If true, the loss will
        be printed. Defaulted to False.
      use_tpr_fpr_for_val: Whether to use TPR@FPR=1% as metric for validation.
        If false, use cross entropy loss.

    Returns:
      Tuple of
        training_history: Training history keyed by epoch number where the
        values are
          dictionaries containing the loss, validation loss, and model
          parameters,
          keyed by
          'loss', 'val_loss', and 'params', respectively.
        min_val_loss: Minimum validation loss achieved during training.
    """

    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Shuffle the data if required
    if shuffle:
        indices = torch.randperm(len(g_values))
        g_values = g_values[indices]
        mask = mask[indices]
        watermarked = watermarked[indices]

    # Initialize optimizer
    optimizer = torch.optim.Adam(detector.parameters(), lr=learning_rate)
    history = {}
    min_val_loss = float("inf")

    for epoch in range(epochs):
        losses = []
        detector.train()
        num_batches = len(g_values) // minibatch_size
        for i in range(0, len(g_values), minibatch_size):
            end = i + minibatch_size
            if end > len(g_values):
                break
            loss_batch_weight = l2_weight / num_batches

            optimizer.zero_grad()
            loss = detector(
                g_values=g_values[i:end],
                mask=mask[i:end],
                labels=watermarked[i:end],
                loss_batch_weight=loss_batch_weight,
            )[1]
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = sum(losses) / len(losses)

        val_losses = []
        if g_values_val is not None:
            detector.eval()
            if validation_metric == ValidationMetric.TPR_AT_FPR:
                val_loss = update_fn_if_fpr_tpr(
                    detector,
                    g_values_val,
                    mask_val,
                    watermarked_val,
                    minibatch_size=minibatch_size,
                )
            else:
                for i in range(0, len(g_values_val), minibatch_size):
                    end = i + minibatch_size
                    if end > len(g_values_val):
                        break
                    with torch.no_grad():
                        v_loss = detector(
                            g_values=g_values_val[i:end],
                            mask=mask_val[i:end],
                            labels=watermarked_val[i:end],
                            loss_batch_weight=0,
                        )[1]
                    val_losses.append(v_loss.item())
                val_loss = sum(val_losses) / len(val_losses)

        # Store training history
        history[epoch + 1] = {"loss": train_loss, "val_loss": val_loss}
        if verbose:
            if val_loss is not None:
                print(f"Epoch {epoch}: loss {loss} (train), {val_loss} (val)")
            else:
                print(f"Epoch {epoch}: loss {loss} (train)")

        if val_loss is not None and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_val_epoch = epoch

    if verbose:
        print(f"Best val Epoch: {best_val_epoch}, min_val_loss: {min_val_loss}")

    return history, min_val_loss


def train_best_detector(
    tokenized_wm_outputs: Union[List[np.ndarray], np.ndarray],
    tokenized_uwm_outputs: Union[List[np.ndarray], np.ndarray],
    logits_processor: SynthIDTextWatermarkLogitsProcessor,
    tokenizer: Any,
    torch_device: torch.device,
    test_size: float = 0.3,
    pos_truncation_length: Optional[int] = 200,
    neg_truncation_length: Optional[int] = 100,
    max_padded_length: int = 2300,
    n_epochs: int = 50,
    learning_rate: float = 2.1e-2,
    l2_weights: np.ndarray = np.logspace(-3, -2, num=4),
    verbose: bool = False,
    validation_metric: ValidationMetric = ValidationMetric.TPR_AT_FPR,
):
    """Train and return the best detector given range of hyperparameters.

    In practice, we have found that tuning pos_truncation_length,
    neg_truncation_length, n_epochs, learning_rate and l2_weights can help
    improve the performance of the detector. We reccommend tuning these
    parameters for your data.
    """
    l2_weights = list(l2_weights)

    (
        train_g_values,
        train_masks,
        train_labels,
        cv_g_values,
        cv_masks,
        cv_labels,
    ) = process_raw_model_outputs(
        logits_processor,
        tokenizer,
        pos_truncation_length,
        neg_truncation_length,
        max_padded_length,
        tokenized_wm_outputs,
        test_size,
        tokenized_uwm_outputs,
        torch_device,
    )

    best_detector = None
    lowest_loss = float("inf")
    val_losses = []
    for l2_weight in l2_weights:
        config = BayesianDetectorConfig(watermarking_depth=len(logits_processor.keys))
        detector = BayesianDetectorModel(config).to(torch_device)
        _, min_val_loss = train_detector(
            detector=detector,
            g_values=train_g_values,
            mask=train_masks,
            watermarked=train_labels,
            g_values_val=cv_g_values,
            mask_val=cv_masks,
            watermarked_val=cv_labels,
            learning_rate=learning_rate,
            l2_weight=l2_weight,
            epochs=n_epochs,
            verbose=verbose,
            validation_metric=validation_metric,
        )
        val_losses.append(min_val_loss)
        if min_val_loss < lowest_loss:
            lowest_loss = min_val_loss
            best_detector = detector
    return best_detector, lowest_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2b-it",
        help=("LM model to train the detector for."),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=("Temperature to sample from the model."),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help=("Top K for sampling."),
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help=("Top P for sampling."),
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=10000,
        help=("Number of negatives for detector training."),
    )
    parser.add_argument(
        "--pos_batch_size",
        type=int,
        default=32,
        help=("Batch size of watermarked positives while sampling."),
    )
    parser.add_argument(
        "--num_pos_batch",
        type=int,
        default=313,
        help=("Number of positive batches for training."),
    )
    parser.add_argument(
        "--generation_length",
        type=int,
        default=512,
        help=("Generation length for sampling."),
    )
    parser.add_argument(
        "--save_model_to_hf_hub",
        action="store_true",
        help=("Whether to save the trained model HF hub. By default it will be a private repo."),
    )
    parser.add_argument(
        "--load_from_hf_hub",
        action="store_true",
        help=(
            "Whether to load trained detector model from HF Hub, make sure its the model trained on the same model "
            "we are loading in the script."
        ),
    )
    parser.add_argument(
        "--hf_hub_model_name",
        type=str,
        default=None,
        help=("HF hub model name for loading of saving the model."),
    )
    parser.add_argument(
        "--eval_detector_on_prompts",
        action="store_true",
        help=("Evaluate detector on a prompt and print probability of watermark."),
    )

    args = parser.parse_args()
    model_name = args.model_name
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    num_negatives = args.num_negatives
    pos_batch_size = args.pos_batch_size
    num_pos_batch = args.num_pos_batch
    if num_pos_batch < 10:
        raise ValueError("--num_pos_batch should be greater than 10.")
    generation_length = args.generation_length
    save_model_to_hf_hub = args.save_model_to_hf_hub
    load_from_hf_hub = args.load_from_hf_hub
    repo_name = args.hf_hub_model_name
    eval_detector_on_prompts = args.eval_detector_on_prompts

    NEG_BATCH_SIZE = 32

    # Truncate outputs to this length for training.
    POS_TRUNCATION_LENGTH = 200
    NEG_TRUNCATION_LENGTH = 100
    # Pad trucated outputs to this length for equal shape across all batches.
    MAX_PADDED_LENGTH = 1000

    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if DEVICE.type not in ("cuda", "tpu"):
        raise ValueError("We have found the training stable on GPU and TPU, we are working on" " a fix for CPUs")

    model = None
    if not load_from_hf_hub:
        # Change this to make your watermark unique. Check documentation in the paper to understand the
        # impact of these parameters.
        DEFAULT_WATERMARKING_CONFIG = {
            "ngram_len": 5,  # This corresponds to H=4 context window size in the paper.
            "keys": [
                654,
                400,
                836,
                123,
                340,
                443,
                597,
                160,
                57,
                29,
                590,
                639,
                13,
                715,
                468,
                990,
                966,
                226,
                324,
                585,
                118,
                504,
                421,
                521,
                129,
                669,
                732,
                225,
                90,
                960,
            ],
            "sampling_table_size": 2**16,
            "sampling_table_seed": 0,
            "context_history_size": 1024,
        }
        watermark_config = SynthIDTextWatermarkingConfig(**DEFAULT_WATERMARKING_CONFIG)

        model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        logits_processor = SynthIDTextWatermarkLogitsProcessor(**DEFAULT_WATERMARKING_CONFIG, device=DEVICE)
        tokenized_wm_outputs = get_tokenized_wm_outputs(
            model,
            tokenizer,
            watermark_config,
            num_pos_batch,
            pos_batch_size,
            temperature,
            generation_length,
            top_k,
            top_p,
            DEVICE,
        )
        tokenized_uwm_outputs = get_tokenized_uwm_outputs(num_negatives, NEG_BATCH_SIZE, tokenizer, DEVICE)

        best_detector, lowest_loss = train_best_detector(
            tokenized_wm_outputs=tokenized_wm_outputs,
            tokenized_uwm_outputs=tokenized_uwm_outputs,
            logits_processor=logits_processor,
            tokenizer=tokenizer,
            torch_device=DEVICE,
            test_size=0.3,
            pos_truncation_length=POS_TRUNCATION_LENGTH,
            neg_truncation_length=NEG_TRUNCATION_LENGTH,
            max_padded_length=MAX_PADDED_LENGTH,
            n_epochs=100,
            learning_rate=3e-3,
            l2_weights=[
                0,
            ],
            verbose=True,
            validation_metric=ValidationMetric.TPR_AT_FPR,
        )
    else:
        if repo_name is None:
            raise ValueError("When loading from pretrained detector model name cannot be None.")
        best_detector = BayesianDetectorModel.from_pretrained(repo_name).to(DEVICE)

    best_detector.config.set_detector_information(
        model_name=model_name, watermarking_config=DEFAULT_WATERMARKING_CONFIG
    )
    if save_model_to_hf_hub:
        upload_model_to_hf(best_detector, repo_name)

    # Evaluate model response with the detector
    if eval_detector_on_prompts:
        model_name = best_detector.config.model_name
        watermark_config_dict = best_detector.config.watermarking_config
        logits_processor = SynthIDTextWatermarkLogitsProcessor(**watermark_config_dict, device=DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        synthid_text_detector = SynthIDTextWatermarkDetector(best_detector, logits_processor, tokenizer)

        if model is None:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
        watermarking_config = SynthIDTextWatermarkingConfig(**watermark_config_dict)

        prompts = ["Write a essay on cats."]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)

        _, inputs_len = inputs["input_ids"].shape

        outputs = model.generate(
            **inputs,
            watermarking_config=watermarking_config,
            do_sample=True,
            max_length=inputs_len + generation_length,
            temperature=temperature,
            top_k=40,
            top_p=1.0,
        )
        outputs = outputs[:, inputs_len:]
        result = synthid_text_detector(outputs)

        # You should set this based on expected fpr (false positive rate) and tpr (true positive rate).
        # Check our demo at HF Spaces for more info.
        upper_threshold = 0.95
        lower_threshold = 0.12
        if result[0][0] > upper_threshold:
            print("The text is watermarked.")
        elif lower_threshold < result[0][0] < upper_threshold:
            print("It is hard to determine if the text is watermarked or not.")
        else:
            print("The text is not watermarked.")
