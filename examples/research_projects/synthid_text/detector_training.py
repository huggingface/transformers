import enum
import gc
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import tqdm
from sklearn import model_selection
from synthid_text import logits_processing, synthid_mixin

import transformers
from transformers import BayesianDetectorConfig, BayesianDetectorModel


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
    truncation_length: int,
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
    outputs = outputs[:, :truncation_length]
    truncation_mask = torch.sum(eos_token_mask, dim=1) >= truncation_length
    return outputs[truncation_mask, :]


def process_outputs_for_training(
    all_outputs: Sequence[torch.Tensor],
    logits_processor: logits_processing.SynthIDLogitsProcessor,
    tokenizer: Any,
    *,
    truncation_length: int,
    max_length: int,
    is_cv: bool,
    is_pos: bool,
    torch_device: torch.device,
) -> tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
    """Process raw model outputs into format understandable by the detector.

    Args:
    all_outputs: sequence of outputs of shape [batch_size, output_len].
    logits_processor: logits processor used for watermarking.
    tokenizer: tokenizer used for the model.
    truncation_length: Length to truncate the outputs.
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
            outputs = filter_and_truncate(outputs, truncation_length, eos_token_mask)

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
    """Calculates TPR at FPR=target_fpr."""
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


@enum.unique
class ValidationMetric(enum.Enum):
    """Direction along the z-axis."""

    TPR_AT_FPR = "tpr_at_fpr"
    CROSS_ENTROPY = "cross_entropy"


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
    g_values_val: torch.Tensor | None = None,
    mask_val: torch.Tensor | None = None,
    watermarked_val: torch.Tensor | None = None,
    verbose: bool = False,
    validation_metric: ValidationMetric = ValidationMetric.TPR_AT_FPR,
) -> tuple[Mapping, float]:
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


def process_raw_model_outputs(
    logits_processor,
    tokenizer,
    truncation_length,
    max_padded_length,
    tokenized_wm_outputs,
    test_size,
    tokenized_uwm_outputs,
    torch_device,
):
    # Split data into train and CV
    train_wm_outputs, cv_wm_outputs = model_selection.train_test_split(tokenized_wm_outputs, test_size=test_size)

    train_uwm_outputs, cv_uwm_outputs = model_selection.train_test_split(tokenized_uwm_outputs, test_size=test_size)

    # Process both train and CV data for training
    wm_masks_train, wm_g_values_train = process_outputs_for_training(
        [torch.tensor(outputs, device=torch_device, dtype=torch.long) for outputs in train_wm_outputs],
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        truncation_length=truncation_length,
        max_length=max_padded_length,
        is_pos=True,
        is_cv=False,
        torch_device=torch_device,
    )
    wm_masks_cv, wm_g_values_cv = process_outputs_for_training(
        [torch.tensor(outputs, device=torch_device, dtype=torch.long) for outputs in cv_wm_outputs],
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        truncation_length=truncation_length,
        max_length=max_padded_length,
        is_pos=True,
        is_cv=True,
        torch_device=torch_device,
    )
    uwm_masks_train, uwm_g_values_train = process_outputs_for_training(
        [torch.tensor(outputs, device=torch_device, dtype=torch.long) for outputs in train_uwm_outputs],
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        truncation_length=truncation_length,
        max_length=max_padded_length,
        is_pos=False,
        is_cv=False,
        torch_device=torch_device,
    )
    uwm_masks_cv, uwm_g_values_cv = process_outputs_for_training(
        [torch.tensor(outputs, device=torch_device, dtype=torch.long) for outputs in cv_uwm_outputs],
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        truncation_length=truncation_length,
        max_length=max_padded_length,
        is_pos=False,
        is_cv=True,
        torch_device=torch_device,
    )

    # We get list of data; here we concat all together to be passed to the
    # detector.
    wm_masks_train = torch.cat(wm_masks_train, dim=0)
    wm_g_values_train = torch.cat(wm_g_values_train, dim=0)
    # Note: Use float instead of bool. Otherwise, the entropy calculation doesn't work
    wm_labels_train = torch.ones((wm_masks_train.shape[0],), dtype=torch.float, device=torch_device)

    wm_masks_cv = torch.cat(wm_masks_cv, dim=0)
    wm_g_values_cv = torch.cat(wm_g_values_cv, dim=0)
    wm_labels_cv = torch.ones((wm_masks_cv.shape[0],), dtype=torch.float, device=torch_device)

    uwm_masks_train = torch.cat(uwm_masks_train, dim=0)
    uwm_g_values_train = torch.cat(uwm_g_values_train, dim=0)
    uwm_labels_train = torch.zeros((uwm_masks_train.shape[0],), dtype=torch.float, device=torch_device)
    uwm_masks_cv = torch.cat(uwm_masks_cv, dim=0)
    uwm_g_values_cv = torch.cat(uwm_g_values_cv, dim=0)
    uwm_labels_cv = torch.zeros((uwm_masks_cv.shape[0],), dtype=torch.float, device=torch_device)

    # Concat pos and negatives data together.
    train_g_values = torch.cat((wm_g_values_train, uwm_g_values_train), dim=0)
    train_labels = torch.cat((wm_labels_train, uwm_labels_train), axis=0)
    train_masks = torch.cat((wm_masks_train, uwm_masks_train), axis=0)

    cv_g_values = torch.cat((wm_g_values_cv, uwm_g_values_cv), axis=0)
    cv_labels = torch.cat((wm_labels_cv, uwm_labels_cv), axis=0)
    cv_masks = torch.cat((wm_masks_cv, uwm_masks_cv), axis=0)

    # Shuffle data.
    train_g_values = train_g_values.squeeze()
    train_labels = train_labels.squeeze()
    train_masks = train_masks.squeeze()

    cv_g_values = cv_g_values.squeeze()
    cv_labels = cv_labels.squeeze()
    cv_masks = cv_masks.squeeze()

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


def train_best_detector(
    tokenized_wm_outputs: Sequence[np.ndarray] | np.ndarray,
    tokenized_uwm_outputs: Sequence[np.ndarray] | np.ndarray,
    logits_processor: logits_processing.SynthIDLogitsProcessor,
    tokenizer: Any,
    torch_device: torch.device,
    test_size: float = 0.3,
    truncation_length: int = 200,
    max_padded_length: int = 2300,
    n_epochs: int = 50,
    learning_rate: float = 2.1e-2,
    l2_weights: np.ndarray = np.logspace(-3, -2, num=4),
    verbose: bool = False,
    validation_metric: ValidationMetric = ValidationMetric.TPR_AT_FPR,
):
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
        truncation_length,
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
    model_name = "google/gemma-7b-it"  # @param ['gpt2', 'google/gemma-2b-it', 'google/gemma-7b-it']

    CONFIG = synthid_mixin.DEFAULT_WATERMARKING_CONFIG
    TEMPERATURE = 0.5
    TOP_K = 40

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logits_processor = logits_processing.SynthIDLogitsProcessor(**CONFIG, top_k=TOP_K, temperature=TEMPERATURE)

    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    NUM_NEGATIVES = 10000
    POS_BATCH_SIZE = 32
    NUM_POS_BATCHES = 313
    NEG_BATCH_SIZE = 32

    # Truncate outputs to this length for training.
    TRUNCATION_LENGTH = 200
    # Pad trucated outputs to this length for equal shape across all batches.
    MAX_PADDED_LENGTH = 2300

    ### CHANGE PATH
    tokenized_wm_outputs = torch.load("/home/marc/local_test/synthia_test/watermarked_outputs_7B_0.7_20k.pkl")
    tokenized_wm_outputs = tokenized_wm_outputs[:10]

    tokenized_uwm_outputs = torch.load("/home/marc/local_test/synthia_test/unwatermarked_human_text.pkl")
    tokenized_uwm_outputs = tokenized_uwm_outputs[:10]

    best_detector, lowest_loss = train_best_detector(
        tokenized_wm_outputs=tokenized_wm_outputs,
        tokenized_uwm_outputs=tokenized_uwm_outputs,
        logits_processor=logits_processor,
        tokenizer=tokenizer,
        torch_device=DEVICE,
        test_size=0.3,
        truncation_length=TRUNCATION_LENGTH,
        max_padded_length=MAX_PADDED_LENGTH,
        n_epochs=50,
        learning_rate=2.1e-2,
        l2_weights=np.logspace(-3, -2, num=4),
        verbose=True,
        validation_metric=ValidationMetric.TPR_AT_FPR,
    )

    best_detector.save_pretrained("bayesian_detector")
