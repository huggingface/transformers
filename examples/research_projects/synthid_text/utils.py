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

import gc
from typing import Any, List, Optional, Tuple

import datasets
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import tqdm
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn import model_selection

import transformers


def pad_to_len(
    arr: torch.Tensor,
    target_len: int,
    left_pad: bool,
    eos_token: int,
    device: torch.device,
) -> torch.Tensor:
    """Pad or truncate array to given length."""
    if arr.shape[1] < target_len:
        shape_for_ones = list(arr.shape)
        shape_for_ones[1] = target_len - shape_for_ones[1]
        padded = (
            torch.ones(
                shape_for_ones,
                device=device,
                dtype=torch.long,
            )
            * eos_token
        )
        if not left_pad:
            arr = torch.concatenate((arr, padded), dim=1)
        else:
            arr = torch.concatenate((padded, arr), dim=1)
    else:
        arr = arr[:, :target_len]
    return arr


def filter_and_truncate(
    outputs: torch.Tensor,
    truncation_length: Optional[int],
    eos_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Filter and truncate outputs to given length.

    Args:
    outputs: output tensor of shape [batch_size, output_len]
    truncation_length: Length to truncate the final output.
    eos_token_mask: EOS token mask of shape [batch_size, output_len]

    Returns:
    output tensor of shape [batch_size, truncation_length].
    """
    if truncation_length:
        outputs = outputs[:, :truncation_length]
        truncation_mask = torch.sum(eos_token_mask, dim=1) >= truncation_length
        return outputs[truncation_mask, :]
    return outputs


def process_outputs_for_training(
    all_outputs: List[torch.Tensor],
    logits_processor: transformers.generation.SynthIDTextWatermarkLogitsProcessor,
    tokenizer: Any,
    pos_truncation_length: Optional[int],
    neg_truncation_length: Optional[int],
    max_length: int,
    is_cv: bool,
    is_pos: bool,
    torch_device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Process raw model outputs into format understandable by the detector.

    Args:
    all_outputs: sequence of outputs of shape [batch_size, output_len].
    logits_processor: logits processor used for watermarking.
    tokenizer: tokenizer used for the model.
    pos_truncation_length: Length to truncate wm outputs.
    neg_truncation_length: Length to truncate uwm outputs.
    max_length: Length to pad truncated outputs so that all processed entries.
        have same shape.
    is_cv: Process given outputs for cross validation.
    is_pos: Process given outputs for positives.
    torch_device: torch device to use.

    Returns:
    Tuple of
        all_masks: list of masks of shape [batch_size, max_length].
        all_g_values: list of g_values of shape [batch_size, max_length, depth].
    """
    all_masks = []
    all_g_values = []
    for outputs in tqdm.tqdm(all_outputs):
        # outputs is of shape [batch_size, output_len].
        # output_len can differ from batch to batch.
        eos_token_mask = logits_processor.compute_eos_token_mask(
            input_ids=outputs,
            eos_token_id=tokenizer.eos_token_id,
        )
        if is_pos or is_cv:
            # filter with length for positives for both train and CV.
            # We also filter for length when CV negatives are processed.
            outputs = filter_and_truncate(outputs, pos_truncation_length, eos_token_mask)
        elif not is_pos and not is_cv:
            outputs = filter_and_truncate(outputs, neg_truncation_length, eos_token_mask)

        # If no filtered outputs skip this batch.
        if outputs.shape[0] == 0:
            continue

        # All outputs are padded to max-length with eos-tokens.
        outputs = pad_to_len(outputs, max_length, False, tokenizer.eos_token_id, torch_device)
        # outputs shape [num_filtered_entries, max_length]

        eos_token_mask = logits_processor.compute_eos_token_mask(
            input_ids=outputs,
            eos_token_id=tokenizer.eos_token_id,
        )

        context_repetition_mask = logits_processor.compute_context_repetition_mask(
            input_ids=outputs,
        )

        # context_repetition_mask of shape [num_filtered_entries, max_length -
        # (ngram_len - 1)].
        context_repetition_mask = pad_to_len(context_repetition_mask, max_length, True, 0, torch_device)
        # We pad on left to get same max_length shape.
        # context_repetition_mask of shape [num_filtered_entries, max_length].
        combined_mask = context_repetition_mask * eos_token_mask

        g_values = logits_processor.compute_g_values(
            input_ids=outputs,
        )

        # g_values of shape [num_filtered_entries, max_length - (ngram_len - 1),
        # depth].
        g_values = pad_to_len(g_values, max_length, True, 0, torch_device)

        # We pad on left to get same max_length shape.
        # g_values of shape [num_filtered_entries, max_length, depth].
        all_masks.append(combined_mask)
        all_g_values.append(g_values)
    return all_masks, all_g_values


def tpr_at_fpr(detector, detector_inputs, w_true, minibatch_size, target_fpr=0.01) -> torch.Tensor:
    """Calculates true positive rate (TPR) at false positive rate (FPR)=target_fpr."""
    positive_idxs = w_true == 1
    negative_idxs = w_true == 0
    num_samples = detector_inputs[0].size(0)

    w_preds = []
    for start in range(0, num_samples, minibatch_size):
        end = start + minibatch_size
        detector_inputs_ = (
            detector_inputs[0][start:end],
            detector_inputs[1][start:end],
        )
        with torch.no_grad():
            w_pred = detector(*detector_inputs_)[0]
        w_preds.append(w_pred)

    w_pred = torch.cat(w_preds, dim=0)  # Concatenate predictions
    positive_scores = w_pred[positive_idxs]
    negative_scores = w_pred[negative_idxs]

    # Calculate the FPR threshold
    # Note: percentile -> quantile
    fpr_threshold = torch.quantile(negative_scores, 1 - target_fpr)
    # Note: need to switch to FP32 since torch.mean doesn't work with torch.bool
    return torch.mean((positive_scores >= fpr_threshold).to(dtype=torch.float32)).item()  # TPR


def update_fn_if_fpr_tpr(detector, g_values_val, mask_val, watermarked_val, minibatch_size):
    """Loss function for negative TPR@FPR=1% as the validation loss."""
    tpr_ = tpr_at_fpr(
        detector=detector,
        detector_inputs=(g_values_val, mask_val),
        w_true=watermarked_val,
        minibatch_size=minibatch_size,
    )
    return -tpr_


def process_raw_model_outputs(
    logits_processor,
    tokenizer,
    pos_truncation_length,
    neg_truncation_length,
    max_padded_length,
    tokenized_wm_outputs,
    test_size,
    tokenized_uwm_outputs,
    torch_device,
):
    # Split data into train and CV
    train_wm_outputs, cv_wm_outputs = model_selection.train_test_split(tokenized_wm_outputs, test_size=test_size)

    train_uwm_outputs, cv_uwm_outputs = model_selection.train_test_split(tokenized_uwm_outputs, test_size=test_size)

    process_kwargs = {
        "logits_processor": logits_processor,
        "tokenizer": tokenizer,
        "pos_truncation_length": pos_truncation_length,
        "neg_truncation_length": neg_truncation_length,
        "max_length": max_padded_length,
        "torch_device": torch_device,
    }

    # Process both train and CV data for training
    wm_masks_train, wm_g_values_train = process_outputs_for_training(
        [torch.tensor(outputs, device=torch_device, dtype=torch.long) for outputs in train_wm_outputs],
        is_pos=True,
        is_cv=False,
        **process_kwargs,
    )
    wm_masks_cv, wm_g_values_cv = process_outputs_for_training(
        [torch.tensor(outputs, device=torch_device, dtype=torch.long) for outputs in cv_wm_outputs],
        is_pos=True,
        is_cv=True,
        **process_kwargs,
    )
    uwm_masks_train, uwm_g_values_train = process_outputs_for_training(
        [torch.tensor(outputs, device=torch_device, dtype=torch.long) for outputs in train_uwm_outputs],
        is_pos=False,
        is_cv=False,
        **process_kwargs,
    )
    uwm_masks_cv, uwm_g_values_cv = process_outputs_for_training(
        [torch.tensor(outputs, device=torch_device, dtype=torch.long) for outputs in cv_uwm_outputs],
        is_pos=False,
        is_cv=True,
        **process_kwargs,
    )

    # We get list of data; here we concat all together to be passed to the detector.
    def pack(mask, g_values):
        mask = torch.cat(mask, dim=0)
        g = torch.cat(g_values, dim=0)
        return mask, g

    wm_masks_train, wm_g_values_train = pack(wm_masks_train, wm_g_values_train)
    # Note: Use float instead of bool. Otherwise, the entropy calculation doesn't work
    wm_labels_train = torch.ones((wm_masks_train.shape[0],), dtype=torch.float, device=torch_device)

    wm_masks_cv, wm_g_values_cv = pack(wm_masks_cv, wm_g_values_cv)
    wm_labels_cv = torch.ones((wm_masks_cv.shape[0],), dtype=torch.float, device=torch_device)

    uwm_masks_train, uwm_g_values_train = pack(uwm_masks_train, uwm_g_values_train)
    uwm_labels_train = torch.zeros((uwm_masks_train.shape[0],), dtype=torch.float, device=torch_device)

    uwm_masks_cv, uwm_g_values_cv = pack(uwm_masks_cv, uwm_g_values_cv)
    uwm_labels_cv = torch.zeros((uwm_masks_cv.shape[0],), dtype=torch.float, device=torch_device)

    # Concat pos and negatives data together.
    train_g_values = torch.cat((wm_g_values_train, uwm_g_values_train), dim=0).squeeze()
    train_labels = torch.cat((wm_labels_train, uwm_labels_train), axis=0).squeeze()
    train_masks = torch.cat((wm_masks_train, uwm_masks_train), axis=0).squeeze()

    cv_g_values = torch.cat((wm_g_values_cv, uwm_g_values_cv), axis=0).squeeze()
    cv_labels = torch.cat((wm_labels_cv, uwm_labels_cv), axis=0).squeeze()
    cv_masks = torch.cat((wm_masks_cv, uwm_masks_cv), axis=0).squeeze()

    # Shuffle data.
    shuffled_idx = torch.randperm(train_g_values.shape[0])  # Use torch for GPU compatibility

    train_g_values = train_g_values[shuffled_idx]
    train_labels = train_labels[shuffled_idx]
    train_masks = train_masks[shuffled_idx]

    # Shuffle the cross-validation data
    shuffled_idx_cv = torch.randperm(cv_g_values.shape[0])  # Use torch for GPU compatibility
    cv_g_values = cv_g_values[shuffled_idx_cv]
    cv_labels = cv_labels[shuffled_idx_cv]
    cv_masks = cv_masks[shuffled_idx_cv]

    # Del some variables so we free up GPU memory.
    del (
        wm_g_values_train,
        wm_labels_train,
        wm_masks_train,
        wm_g_values_cv,
        wm_labels_cv,
        wm_masks_cv,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return train_g_values, train_masks, train_labels, cv_g_values, cv_masks, cv_labels


def get_tokenized_uwm_outputs(num_negatives, neg_batch_size, tokenizer, device):
    dataset, info = tfds.load("wikipedia/20230601.en", split="train", with_info=True)
    dataset = dataset.take(num_negatives)

    # Convert the dataset to a DataFrame
    df = tfds.as_dataframe(dataset, info)
    ds = tf.data.Dataset.from_tensor_slices(dict(df))
    tf.random.set_seed(0)
    ds = ds.shuffle(buffer_size=10_000)
    ds = ds.batch(batch_size=neg_batch_size)

    tokenized_uwm_outputs = []
    # Pad to this length (on the right) for batching.
    padded_length = 1000
    for i, batch in tqdm.tqdm(enumerate(ds)):
        responses = [val.decode() for val in batch["text"].numpy()]
        inputs = tokenizer(
            responses,
            return_tensors="pt",
            padding=True,
        ).to(device)
        inputs = inputs["input_ids"].cpu().numpy()
        if inputs.shape[1] >= padded_length:
            inputs = inputs[:, :padded_length]
        else:
            inputs = np.concatenate(
                [inputs, np.ones((neg_batch_size, padded_length - inputs.shape[1])) * tokenizer.eos_token_id], axis=1
            )
        tokenized_uwm_outputs.append(inputs)
        if len(tokenized_uwm_outputs) * neg_batch_size > num_negatives:
            break
    return tokenized_uwm_outputs


def get_tokenized_wm_outputs(
    model,
    tokenizer,
    watermark_config,
    num_pos_batches,
    pos_batch_size,
    temperature,
    max_output_len,
    top_k,
    top_p,
    device,
):
    eli5_prompts = datasets.load_dataset("Pavithree/eli5")

    wm_outputs = []

    for batch_id in tqdm.tqdm(range(num_pos_batches)):
        prompts = eli5_prompts["train"]["title"][batch_id * pos_batch_size : (batch_id + 1) * pos_batch_size]
        prompts = [prompt.strip('"') for prompt in prompts]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(device)
        _, inputs_len = inputs["input_ids"].shape

        outputs = model.generate(
            **inputs,
            watermarking_config=watermark_config,
            do_sample=True,
            max_length=inputs_len + max_output_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        wm_outputs.append(outputs[:, inputs_len:].cpu().detach())

        del outputs, inputs, prompts
        gc.collect()

    gc.collect()
    torch.cuda.empty_cache()
    return wm_outputs


def upload_model_to_hf(model, hf_repo_name: str, private: bool = True):
    api = HfApi()

    # Check if the repository exists
    try:
        api.repo_info(repo_id=hf_repo_name, use_auth_token=True)
        print(f"Repository '{hf_repo_name}' already exists.")
    except RepositoryNotFoundError:
        # If the repository does not exist, create it
        print(f"Repository '{hf_repo_name}' not found. Creating it...")
        create_repo(repo_id=hf_repo_name, private=private, use_auth_token=True)
        print(f"Repository '{hf_repo_name}' created successfully.")

    # Push the model to the Hugging Face Hub
    print(f"Uploading model to Hugging Face repo '{hf_repo_name}'...")
    model.push_to_hub(repo_id=hf_repo_name, use_auth_token=True)
