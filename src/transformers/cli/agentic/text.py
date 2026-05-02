# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""
Text inference CLI commands.

Each function uses Auto* model and tokenizer classes directly and is
registered as a top-level ``transformers`` CLI command via ``app.py``.
"""

from typing import Annotated

import typer

from ._common import (
    DeviceOpt,
    DtypeOpt,
    JsonOpt,
    ModelOpt,
    RevisionOpt,
    TokenOpt,
    TrustOpt,
    _load_pretrained,
    format_output,
    resolve_input,
)


def _aggregate_entities(entities, text):
    """Merge sub-word entity predictions into whole entities (B-/I- tag merging)."""
    if not entities:
        return entities

    aggregated = []
    current = None

    for entity in entities:
        label = entity["entity"]
        entity_type = label.split("-", 1)[-1] if "-" in label else label
        is_continuation = label.startswith("I-")

        if current is not None and is_continuation and entity_type == current["entity_group"]:
            current["end"] = entity["end"]
            current["score"] = min(current["score"], entity["score"])
        else:
            if current is not None:
                current["word"] = text[current["start"] : current["end"]]
                aggregated.append(current)
            current = {
                "entity_group": entity_type,
                "score": entity["score"],
                "start": entity["start"],
                "end": entity["end"],
            }

    if current is not None:
        current["word"] = text[current["start"] : current["end"]]
        aggregated.append(current)

    return aggregated


def classify(
    text: Annotated[str | None, typer.Option(help="Text to classify.")] = None,
    file: Annotated[str | None, typer.Option(help="Read text from this file.")] = None,
    labels: Annotated[
        str | None, typer.Option(help="Comma-separated candidate labels for zero-shot classification.")
    ] = None,
    model: ModelOpt = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Classify text into categories.

    Uses ``AutoModelForSequenceClassification`` by default (requires a
    fine-tuned classification model). Pass ``--labels`` to switch to
    zero-shot classification via natural language inference.

    Examples::

        # Supervised (model already fine-tuned for sentiment)
        transformers classify --model distilbert/distilbert-base-uncased-finetuned-sst-2-english --text "Great movie!"

        # Zero-shot (any categories, no fine-tuning needed)
        transformers classify --text "The stock market crashed" --labels "politics,finance,sports"

        # Read from file, output as JSON
        transformers classify --file review.txt --json
    """
    import torch

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    input_text = resolve_input(text, file)

    if labels is not None:
        # Zero-shot classification via natural language inference:
        # for each candidate label, test whether the input entails "This example is {label}."
        model_id = model or "facebook/bart-large-mnli"
        loaded_model, tokenizer = _load_pretrained(
            AutoModelForSequenceClassification,
            AutoTokenizer,
            model_id,
            device,
            dtype,
            trust_remote_code,
            token,
            revision,
        )

        candidate_labels = [l.strip() for l in labels.split(",")]

        # Find the entailment class index from the model config
        entail_idx = 2
        for idx, label_name in loaded_model.config.id2label.items():
            if label_name.lower().startswith("entail"):
                entail_idx = int(idx)
                break

        scores = []
        for label in candidate_labels:
            hypothesis = f"This example is {label}."
            inputs = tokenizer(input_text, hypothesis, return_tensors="pt", truncation=True)
            if hasattr(loaded_model, "device"):
                inputs = inputs.to(loaded_model.device)
            with torch.no_grad():
                logits = loaded_model(**inputs).logits
            scores.append(logits.softmax(dim=-1)[0, entail_idx].item())

        total = sum(scores)
        result = {
            "sequence": input_text,
            "labels": candidate_labels,
            "scores": [s / total for s in scores],
        }
    else:
        model_id = model or "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        loaded_model, tokenizer = _load_pretrained(
            AutoModelForSequenceClassification,
            AutoTokenizer,
            model_id,
            device,
            dtype,
            trust_remote_code,
            token,
            revision,
        )

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        if hasattr(loaded_model, "device"):
            inputs = inputs.to(loaded_model.device)
        with torch.no_grad():
            logits = loaded_model(**inputs).logits

        probs = logits.softmax(dim=-1)[0]
        top_idx = probs.argmax().item()
        result = [{"label": loaded_model.config.id2label[top_idx], "score": probs[top_idx].item()}]

    print(format_output(result, output_json))


def ner(
    text: Annotated[str | None, typer.Option(help="Text to extract entities from.")] = None,
    file: Annotated[str | None, typer.Option(help="Read text from this file.")] = None,
    model: ModelOpt = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    aggregation_strategy: Annotated[str, typer.Option(help="Entity aggregation: 'none' or 'simple'.")] = "simple",
    output_json: JsonOpt = False,
):
    """
    Extract named entities from text (NER).

    Uses ``AutoModelForTokenClassification`` with entity aggregation
    enabled by default (``--aggregation-strategy simple``).

    Example::

        transformers ner --model dslim/bert-base-NER --text "Apple CEO Tim Cook met with President Biden in Washington."
    """
    import torch

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    input_text = resolve_input(text, file)
    model_id = model or "dslim/bert-base-NER"
    loaded_model, tokenizer = _load_pretrained(
        AutoModelForTokenClassification, AutoTokenizer, model_id, device, dtype, trust_remote_code, token, revision
    )

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")[0]
    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)

    with torch.no_grad():
        logits = loaded_model(**inputs).logits

    probs = logits.softmax(dim=-1)
    predictions = logits.argmax(dim=-1)[0]

    entities = []
    for idx, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
        label = loaded_model.config.id2label[pred.item()]
        if label == "O" or (start == 0 and end == 0):
            continue
        entities.append(
            {
                "entity": label,
                "score": probs[0, idx, pred].item(),
                "word": input_text[start:end],
                "start": start.item(),
                "end": end.item(),
            }
        )

    if aggregation_strategy == "simple":
        entities = _aggregate_entities(entities, input_text)

    print(format_output(entities, output_json))


def token_classify(
    text: Annotated[str | None, typer.Option(help="Text to tag.")] = None,
    file: Annotated[str | None, typer.Option(help="Read text from this file.")] = None,
    model: ModelOpt = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Tag tokens with labels (POS tagging, chunking, etc.).

    Uses ``AutoModelForTokenClassification``. The output depends on the
    model — a POS model outputs POS tags, a NER model outputs entity labels.

    Example::

        transformers token-classify --model vblagoje/bert-english-uncased-finetuned-pos --text "The cat sat on the mat."
    """
    import torch

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    input_text = resolve_input(text, file)
    model_id = model or "vblagoje/bert-english-uncased-finetuned-pos"
    loaded_model, tokenizer = _load_pretrained(
        AutoModelForTokenClassification, AutoTokenizer, model_id, device, dtype, trust_remote_code, token, revision
    )

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")[0]
    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)

    with torch.no_grad():
        logits = loaded_model(**inputs).logits

    probs = logits.softmax(dim=-1)
    predictions = logits.argmax(dim=-1)[0]

    result = []
    for idx, (pred, (start, end)) in enumerate(zip(predictions, offset_mapping)):
        if start == 0 and end == 0:
            continue
        result.append(
            {
                "entity": loaded_model.config.id2label[pred.item()],
                "score": probs[0, idx, pred].item(),
                "word": input_text[start:end],
                "start": start.item(),
                "end": end.item(),
            }
        )

    print(format_output(result, output_json))


def qa(
    question: Annotated[str, typer.Option(help="The question to answer.")],
    context: Annotated[str | None, typer.Option(help="Context paragraph containing the answer.")] = None,
    file: Annotated[str | None, typer.Option(help="Read context from this file.")] = None,
    model: ModelOpt = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Answer a question given a context paragraph (extractive QA).

    Uses ``AutoModelForQuestionAnswering`` to extract the answer span from
    ``--context`` (or ``--file``). The model does not generate new text —
    it highlights the relevant substring.

    Example::

        transformers qa --question "Who invented the telephone?" --context "Alexander Graham Bell invented the telephone in 1876."
    """
    import torch

    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    ctx = resolve_input(context, file)
    model_id = model or "distilbert/distilbert-base-cased-distilled-squad"
    loaded_model, tokenizer = _load_pretrained(
        AutoModelForQuestionAnswering, AutoTokenizer, model_id, device, dtype, trust_remote_code, token, revision
    )

    inputs = tokenizer(question, ctx, return_tensors="pt", truncation=True)
    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)

    with torch.no_grad():
        outputs = loaded_model(**inputs)

    start_idx = outputs.start_logits.argmax(dim=-1).item()
    end_idx = outputs.end_logits.argmax(dim=-1).item()
    answer_ids = inputs["input_ids"][0, start_idx : end_idx + 1]
    score = (outputs.start_logits[0, start_idx] + outputs.end_logits[0, end_idx]).item()

    result = {
        "answer": tokenizer.decode(answer_ids, skip_special_tokens=True),
        "score": score,
        "start": start_idx,
        "end": end_idx,
    }
    print(format_output(result, output_json))


def table_qa(
    question: Annotated[str, typer.Option(help="Question about the table.")],
    table: Annotated[str, typer.Option(help="Path to a CSV file containing the table.")],
    model: ModelOpt = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Answer a question about tabular data (CSV).

    Loads a CSV file into a table and uses ``AutoModelForTableQuestionAnswering``
    (e.g., TAPAS) to answer the question.

    Example::

        transformers table-qa --question "What is the total revenue?" --table financials.csv
    """
    import pandas as pd
    import torch

    from transformers import AutoModelForTableQuestionAnswering, AutoTokenizer

    model_id = model or "google/tapas-base-finetuned-wtq"
    loaded_model, tokenizer = _load_pretrained(
        AutoModelForTableQuestionAnswering, AutoTokenizer, model_id, device, dtype, trust_remote_code, token, revision
    )

    table_df = pd.read_csv(table).astype(str)
    inputs = tokenizer(table=table_df, queries=question, return_tensors="pt", truncation=True)
    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)

    with torch.no_grad():
        outputs = loaded_model(**inputs)

    logits_agg = getattr(outputs, "logits_aggregation", None)
    if logits_agg is not None:
        predicted_coordinates, predicted_agg = tokenizer.convert_logits_to_predictions(
            inputs, outputs.logits.detach().cpu(), logits_agg.detach().cpu()
        )
        agg_idx = predicted_agg[0]
    else:
        (predicted_coordinates,) = tokenizer.convert_logits_to_predictions(inputs, outputs.logits.detach().cpu())
        agg_idx = 0

    coordinates = predicted_coordinates[0]
    cells = [table_df.iat[row, col] for row, col in coordinates]

    _AGG_OPS = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
    if agg_idx == 1:
        try:
            answer = str(sum(float(c) for c in cells))
        except ValueError:
            answer = ", ".join(cells)
    elif agg_idx == 2:
        try:
            answer = str(sum(float(c) for c in cells) / len(cells))
        except (ValueError, ZeroDivisionError):
            answer = ", ".join(cells)
    elif agg_idx == 3:
        answer = str(len(cells))
    else:
        answer = ", ".join(cells)

    result = {
        "answer": answer,
        "coordinates": coordinates,
        "cells": cells,
        "aggregator": _AGG_OPS.get(agg_idx, "NONE"),
    }
    print(format_output(result, output_json))


def summarize(
    text: Annotated[str | None, typer.Option(help="Text to summarize.")] = None,
    file: Annotated[str | None, typer.Option(help="Read text from this file.")] = None,
    model: ModelOpt = None,
    max_length: Annotated[int | None, typer.Option(help="Maximum summary length in tokens.")] = None,
    min_length: Annotated[int | None, typer.Option(help="Minimum summary length in tokens.")] = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Summarize text.

    Uses ``AutoModelForSeq2SeqLM`` (e.g., BART, T5, Pegasus).

    Examples::

        transformers summarize --model facebook/bart-large-cnn --file article.txt
        transformers summarize --text "Long article text here..." --max-length 100
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    input_text = resolve_input(text, file)
    model_id = model or "facebook/bart-large-cnn"
    loaded_model, tokenizer = _load_pretrained(
        AutoModelForSeq2SeqLM, AutoTokenizer, model_id, device, dtype, trust_remote_code, token, revision
    )

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)

    gen_kwargs = {}
    if max_length is not None:
        gen_kwargs["max_length"] = max_length
    if min_length is not None:
        gen_kwargs["min_length"] = min_length

    output_ids = loaded_model.generate(**inputs, **gen_kwargs)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    result = [{"summary_text": summary}]
    print(format_output(result, output_json))


def translate(
    text: Annotated[str | None, typer.Option(help="Text to translate.")] = None,
    file: Annotated[str | None, typer.Option(help="Read text from this file.")] = None,
    model: ModelOpt = None,
    max_length: Annotated[int | None, typer.Option(help="Maximum translation length.")] = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Translate text between languages.

    Uses ``AutoModelForSeq2SeqLM``. The language pair is determined by the
    model. Use Helsinki-NLP models for specific pairs (e.g.,
    ``Helsinki-NLP/opus-mt-en-de`` for English to German).

    Example::

        transformers translate --model Helsinki-NLP/opus-mt-en-de --text "The weather is nice today."
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    input_text = resolve_input(text, file)
    model_id = model or "Helsinki-NLP/opus-mt-en-de"
    loaded_model, tokenizer = _load_pretrained(
        AutoModelForSeq2SeqLM, AutoTokenizer, model_id, device, dtype, trust_remote_code, token, revision
    )

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)

    gen_kwargs = {}
    if max_length is not None:
        gen_kwargs["max_length"] = max_length

    output_ids = loaded_model.generate(**inputs, **gen_kwargs)
    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    result = [{"translation_text": translation}]
    print(format_output(result, output_json))


def fill_mask(
    text: Annotated[str, typer.Option(help="Text with a [MASK] token.")],
    model: ModelOpt = None,
    top_k: Annotated[int, typer.Option(help="Number of predictions to return.")] = 5,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """
    Predict the masked token in a sentence.

    Uses ``AutoModelForMaskedLM``. The mask token depends on the model
    (``[MASK]`` for BERT, ``<mask>`` for RoBERTa).

    Example::

        transformers fill-mask --model answerdotai/ModernBERT-base --text "The capital of France is [MASK]."
    """
    import torch

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model_id = model or "answerdotai/ModernBERT-base"
    loaded_model, tokenizer = _load_pretrained(
        AutoModelForMaskedLM, AutoTokenizer, model_id, device, dtype, trust_remote_code, token, revision
    )

    inputs = tokenizer(text, return_tensors="pt")
    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)

    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    if len(mask_positions) == 0:
        raise SystemExit(f"No mask token found. Use '{tokenizer.mask_token}' in your text.")

    with torch.no_grad():
        logits = loaded_model(**inputs).logits

    mask_logits = logits[0, mask_positions[0]]
    probs = mask_logits.softmax(dim=-1)
    top_probs, top_ids = probs.topk(top_k)

    result = []
    for prob, token_id in zip(top_probs, top_ids):
        token_str = tokenizer.decode([token_id]).strip()
        result.append(
            {
                "score": prob.item(),
                "token": token_id.item(),
                "token_str": token_str,
                "sequence": text.replace(tokenizer.mask_token, token_str, 1),
            }
        )

    print(format_output(result, output_json))
