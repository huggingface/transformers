# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Doc utilities: Utilities related to documentation
"""

import functools
import inspect
import re
import textwrap
import types
from collections import OrderedDict


def get_docstring_indentation_level(func):
    """Return the indentation level of the start of the docstring of a class or function (or method)."""
    # We assume classes are always defined in the global scope
    if inspect.isclass(func):
        return 4
    source = inspect.getsource(func)
    first_line = source.splitlines()[0]
    function_def_level = len(first_line) - len(first_line.lstrip())
    return 4 + function_def_level


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


def add_start_docstrings_to_model_forward(*docstr):
    def docstring_decorator(fn):
        class_name = f"[`{fn.__qualname__.split('.')[0]}`]"
        intro = rf"""    The {class_name} forward method, overrides the `__call__` special method.

    <Tip>

    Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
    instance afterwards instead of this since the former takes care of running the pre and post processing steps while
    the latter silently ignores them.

    </Tip>
"""

        correct_indentation = get_docstring_indentation_level(fn)
        current_doc = fn.__doc__ if fn.__doc__ is not None else ""
        try:
            first_non_empty = next(line for line in current_doc.splitlines() if line.strip() != "")
            doc_indentation = len(first_non_empty) - len(first_non_empty.lstrip())
        except StopIteration:
            doc_indentation = correct_indentation

        docs = docstr
        # In this case, the correct indentation level (class method, 2 Python levels) was respected, and we should
        # correctly reindent everything. Otherwise, the doc uses a single indentation level
        if doc_indentation == 4 + correct_indentation:
            docs = [textwrap.indent(textwrap.dedent(doc), " " * correct_indentation) for doc in docstr]
            intro = textwrap.indent(textwrap.dedent(intro), " " * correct_indentation)

        docstring = "".join(docs) + current_doc
        fn.__doc__ = intro + docstring
        return fn

    return docstring_decorator


def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "".join(docstr)
        return fn

    return docstring_decorator


PT_RETURN_INTRODUCTION = r"""
    Returns:
        [`{full_output_type}`] or `tuple(torch.FloatTensor)`: A [`{full_output_type}`] or a tuple of
        `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
        elements depending on the configuration ([`{config_class}`]) and inputs.

"""


TF_RETURN_INTRODUCTION = r"""
    Returns:
        [`{full_output_type}`] or `tuple(tf.Tensor)`: A [`{full_output_type}`] or a tuple of `tf.Tensor` (if
        `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the
        configuration ([`{config_class}`]) and inputs.

"""


def _get_indent(t):
    """Returns the indentation in the first line of t"""
    search = re.search(r"^(\s*)\S", t)
    return "" if search is None else search.groups()[0]


def _convert_output_args_doc(output_args_doc):
    """Convert output_args_doc to display properly."""
    # Split output_arg_doc in blocks argument/description
    indent = _get_indent(output_args_doc)
    blocks = []
    current_block = ""
    for line in output_args_doc.split("\n"):
        # If the indent is the same as the beginning, the line is the name of new arg.
        if _get_indent(line) == indent:
            if len(current_block) > 0:
                blocks.append(current_block[:-1])
            current_block = f"{line}\n"
        else:
            # Otherwise it's part of the description of the current arg.
            # We need to remove 2 spaces to the indentation.
            current_block += f"{line[2:]}\n"
    blocks.append(current_block[:-1])

    # Format each block for proper rendering
    for i in range(len(blocks)):
        blocks[i] = re.sub(r"^(\s+)(\S+)(\s+)", r"\1- **\2**\3", blocks[i])
        blocks[i] = re.sub(r":\s*\n\s*(\S)", r" -- \1", blocks[i])

    return "\n".join(blocks)


def _prepare_output_docstrings(output_type, config_class, min_indent=None, add_intro=True):
    """
    Prepares the return part of the docstring using `output_type`.
    """
    output_docstring = output_type.__doc__
    params_docstring = None
    if output_docstring is not None:
        # Remove the head of the docstring to keep the list of args only
        lines = output_docstring.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*(Args|Parameters):\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            params_docstring = "\n".join(lines[(i + 1) :])
            params_docstring = _convert_output_args_doc(params_docstring)
        elif add_intro:
            raise ValueError(
                f"No `Args` or `Parameters` section is found in the docstring of `{output_type.__name__}`. Make sure it has "
                "docstring and contain either `Args` or `Parameters`."
            )

    # Add the return introduction
    if add_intro:
        full_output_type = f"{output_type.__module__}.{output_type.__name__}"
        intro = TF_RETURN_INTRODUCTION if output_type.__name__.startswith("TF") else PT_RETURN_INTRODUCTION
        intro = intro.format(full_output_type=full_output_type, config_class=config_class)
    else:
        full_output_type = str(output_type)
        intro = f"\nReturns:\n    `{full_output_type}`"
        if params_docstring is not None:
            intro += ":\n"

    result = intro
    if params_docstring is not None:
        result += params_docstring

    # Apply minimum indent if necessary
    if min_indent is not None:
        lines = result.split("\n")
        # Find the indent of the first nonempty line
        i = 0
        while len(lines[i]) == 0:
            i += 1
        indent = len(_get_indent(lines[i]))
        # If too small, add indentation to all nonempty lines
        if indent < min_indent:
            to_add = " " * (min_indent - indent)
            lines = [(f"{to_add}{line}" if len(line) > 0 else line) for line in lines]
            result = "\n".join(lines)

    return result


FAKE_MODEL_DISCLAIMER = """
    <Tip warning={true}>

    This example uses a random model as the real ones are all very big. To get proper results, you should use
    {real_checkpoint} instead of {fake_checkpoint}. If you get out-of-memory when loading that checkpoint, you can try
    adding `device_map="auto"` in the `from_pretrained` call.

    </Tip>
"""


PT_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer(
    ...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
    ... )

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> predicted_token_class_ids = logits.argmax(-1)

    >>> # Note that tokens are classified rather then input words which means that
    >>> # there might be more predicted token classes than words.
    >>> # Multiple token classes might account for the same word
    >>> predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
    >>> predicted_tokens_classes
    {expected_output}

    >>> labels = predicted_token_class_ids
    >>> loss = model(**inputs, labels=labels).loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```
"""

PT_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    >>> inputs = tokenizer(question, text, return_tensors="pt")
    >>> with torch.no_grad():
    ...     outputs = model(**inputs)

    >>> answer_start_index = outputs.start_logits.argmax()
    >>> answer_end_index = outputs.end_logits.argmax()

    >>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    >>> tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
    {expected_output}

    >>> # target is "nice puppet"
    >>> target_start_index = torch.tensor([{qa_target_start_index}])
    >>> target_end_index = torch.tensor([{qa_target_end_index}])

    >>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
    >>> loss = outputs.loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```
"""

PT_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example of single-label classification:

    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> predicted_class_id = logits.argmax().item()
    >>> model.config.id2label[predicted_class_id]
    {expected_output}

    >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
    >>> num_labels = len(model.config.id2label)
    >>> model = {model_class}.from_pretrained("{checkpoint}", num_labels=num_labels)

    >>> labels = torch.tensor([1])
    >>> loss = model(**inputs, labels=labels).loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```

    Example of multi-label classification:

    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", problem_type="multi_label_classification")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

    >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
    >>> num_labels = len(model.config.id2label)
    >>> model = {model_class}.from_pretrained(
    ...     "{checkpoint}", num_labels=num_labels, problem_type="multi_label_classification"
    ... )

    >>> labels = torch.sum(
    ...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
    ... ).to(torch.float)
    >>> loss = model(**inputs, labels=labels).loss
    ```
"""

PT_MASKED_LM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> # retrieve index of {mask}
    >>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    >>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    >>> tokenizer.decode(predicted_token_id)
    {expected_output}

    >>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
    >>> # mask labels of non-{mask} tokens
    >>> labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

    >>> outputs = model(**inputs, labels=labels)
    >>> round(outputs.loss.item(), 2)
    {expected_loss}
    ```
"""

PT_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

PT_MULTIPLE_CHOICE_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    >>> choice0 = "It is eaten with a fork and a knife."
    >>> choice1 = "It is eaten while held in the hand."
    >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

    >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
    >>> outputs = model(**{{k: v.unsqueeze(0) for k, v in encoding.items()}}, labels=labels)  # batch size is 1

    >>> # the linear classifier still needs to be trained
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    ```
"""

PT_CAUSAL_LM_SAMPLE = r"""
    Example:

    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs, labels=inputs["input_ids"])
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    ```
"""

PT_SPEECH_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoProcessor, {model_class}
    >>> import torch
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    >>> with torch.no_grad():
    ...     outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    >>> list(last_hidden_states.shape)
    {expected_output}
    ```
"""

PT_SPEECH_CTC_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoProcessor, {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits
    >>> predicted_ids = torch.argmax(logits, dim=-1)

    >>> # transcribe speech
    >>> transcription = processor.batch_decode(predicted_ids)
    >>> transcription[0]
    {expected_output}

    >>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

    >>> # compute loss
    >>> loss = model(**inputs).loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```
"""

PT_SPEECH_SEQ_CLASS_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoFeatureExtractor, {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> predicted_class_ids = torch.argmax(logits, dim=-1).item()
    >>> predicted_label = model.config.id2label[predicted_class_ids]
    >>> predicted_label
    {expected_output}

    >>> # compute loss - target_label is e.g. "down"
    >>> target_label = model.config.id2label[0]
    >>> inputs["labels"] = torch.tensor([model.config.label2id[target_label]])
    >>> loss = model(**inputs).loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```
"""


PT_SPEECH_FRAME_CLASS_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoFeatureExtractor, {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(dataset[0]["audio"]["array"], return_tensors="pt", sampling_rate=sampling_rate)
    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> probabilities = torch.sigmoid(logits[0])
    >>> # labels is a one-hot array of shape (num_frames, num_speakers)
    >>> labels = (probabilities > 0.5).long()
    >>> labels[0].tolist()
    {expected_output}
    ```
"""


PT_SPEECH_XVECTOR_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoFeatureExtractor, {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(
    ...     [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
    ... )
    >>> with torch.no_grad():
    ...     embeddings = model(**inputs).embeddings

    >>> embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

    >>> # the resulting embeddings can be used for cosine similarity-based retrieval
    >>> cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    >>> similarity = cosine_sim(embeddings[0], embeddings[1])
    >>> threshold = 0.7  # the optimal threshold is dataset-dependent
    >>> if similarity < threshold:
    ...     print("Speakers are not the same!")
    >>> round(similarity.item(), 2)
    {expected_output}
    ```
"""

PT_VISION_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoImageProcessor, {model_class}
    >>> import torch
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("huggingface/cats-image")
    >>> image = dataset["test"]["image"][0]

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = image_processor(image, return_tensors="pt")

    >>> with torch.no_grad():
    ...     outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    >>> list(last_hidden_states.shape)
    {expected_output}
    ```
"""

PT_VISION_SEQ_CLASS_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoImageProcessor, {model_class}
    >>> import torch
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("huggingface/cats-image")
    >>> image = dataset["test"]["image"][0]

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = image_processor(image, return_tensors="pt")

    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits

    >>> # model predicts one of the 1000 ImageNet classes
    >>> predicted_label = logits.argmax(-1).item()
    >>> print(model.config.id2label[predicted_label])
    {expected_output}
    ```
"""


PT_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": PT_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": PT_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": PT_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": PT_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": PT_MASKED_LM_SAMPLE,
    "LMHead": PT_CAUSAL_LM_SAMPLE,
    "BaseModel": PT_BASE_MODEL_SAMPLE,
    "SpeechBaseModel": PT_SPEECH_BASE_MODEL_SAMPLE,
    "CTC": PT_SPEECH_CTC_SAMPLE,
    "AudioClassification": PT_SPEECH_SEQ_CLASS_SAMPLE,
    "AudioFrameClassification": PT_SPEECH_FRAME_CLASS_SAMPLE,
    "AudioXVector": PT_SPEECH_XVECTOR_SAMPLE,
    "VisionBaseModel": PT_VISION_BASE_MODEL_SAMPLE,
    "ImageClassification": PT_VISION_SEQ_CLASS_SAMPLE,
}


TEXT_TO_AUDIO_SPECTROGRAM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoProcessor, {model_class}, SpeechT5HifiGan

    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    >>> vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    >>> inputs = processor(text="Hello, my dog is cute", return_tensors="pt")

    >>> # generate speech
    >>> speech = model.generate(inputs["input_ids"], speaker_embeddings=speaker_embeddings, vocoder=vocoder)
    ```
"""


TEXT_TO_AUDIO_WAVEFORM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoProcessor, {model_class}

    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    >>> inputs = processor(text="Hello, my dog is cute", return_tensors="pt")

    >>> # generate speech
    >>> speech = model(inputs["input_ids"])
    ```
"""


AUDIO_FRAME_CLASSIFICATION_SAMPLE = PT_SPEECH_FRAME_CLASS_SAMPLE


AUDIO_XVECTOR_SAMPLE = PT_SPEECH_XVECTOR_SAMPLE


IMAGE_TO_TEXT_SAMPLE = r"""
    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, {model_class}

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> inputs = processor(images=image, return_tensors="pt")

    >>> outputs = model(**inputs)
    ```
"""


DEPTH_ESTIMATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoImageProcessor, {model_class}
    >>> import torch
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> processor = AutoImageProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> model.to(device)

    >>> # prepare image for the model
    >>> inputs = processor(images=image, return_tensors="pt").to(device)

    >>> with torch.no_grad():
    ...     outputs = model(**inputs)

    >>> # interpolate to original size
    >>> post_processed_output = processor.post_process_depth_estimation(
    ...     outputs, [(image.height, image.width)],
    ... )
    >>> predicted_depth = post_processed_output[0]["predicted_depth"]
    ```
"""


VIDEO_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    ```
"""


ZERO_SHOT_OBJECT_DETECTION_SAMPLE = r"""
    Example:

    ```python
    ```
"""


IMAGE_TO_IMAGE_SAMPLE = r"""
    Example:

    ```python
    ```
"""


IMAGE_FEATURE_EXTRACTION_SAMPLE = r"""
    Example:

    ```python
    ```
"""


DOCUMENT_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    ```python
    ```
"""


NEXT_SENTENCE_PREDICTION_SAMPLE = r"""
    Example:

    ```python
    ```
"""


MULTIPLE_CHOICE_SAMPLE = PT_MULTIPLE_CHOICE_SAMPLE


PRETRAINING_SAMPLE = r"""
    Example:

    ```python
    ```
"""
MASK_GENERATION_SAMPLE = r"""
    Example:

    ```python
    ```
"""


VISUAL_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    ```python
    ```
"""


TEXT_GENERATION_SAMPLE = r"""
    Example:

    ```python
    ```
"""


IMAGE_CLASSIFICATION_SAMPLE = PT_VISION_SEQ_CLASS_SAMPLE


IMAGE_SEGMENTATION_SAMPLE = r"""
    Example:

    ```python
    ```
"""


FILL_MASK_SAMPLE = r"""
    Example:

    ```python
    ```
"""


OBJECT_DETECTION_SAMPLE = r"""
    Example:

    ```python
    ```
"""


QUESTION_ANSWERING_SAMPLE = PT_QUESTION_ANSWERING_SAMPLE


TEXT2TEXT_GENERATION_SAMPLE = r"""
    Example:

    ```python
    ```
"""


TEXT_CLASSIFICATION_SAMPLE = PT_SEQUENCE_CLASSIFICATION_SAMPLE


TABLE_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    ```python
    ```
"""


TOKEN_CLASSIFICATION_SAMPLE = PT_TOKEN_CLASSIFICATION_SAMPLE


AUDIO_CLASSIFICATION_SAMPLE = PT_SPEECH_SEQ_CLASS_SAMPLE


AUTOMATIC_SPEECH_RECOGNITION_SAMPLE = PT_SPEECH_CTC_SAMPLE


ZERO_SHOT_IMAGE_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    ```
"""


IMAGE_TEXT_TO_TEXT_GENERATION_SAMPLE = r"""
    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, {model_class}

    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")

    >>> messages = [
    ...     {{
    ...         "role": "user", "content": [
    ...             {{"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"}},
    ...             {{"type": "text", "text": "Where is the cat standing?"}},
    ...         ]
    ...     }},
    ... ]

    >>> inputs = processor.apply_chat_template(
    ...     messages,
    ...     tokenize=True,
    ...     return_dict=True,
    ...     return_tensors="pt",
    ...     add_generation_prompt=True
    ... )
    >>> # Generate
    >>> generate_ids = model.generate(**inputs)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    ```
"""


PIPELINE_TASKS_TO_SAMPLE_DOCSTRINGS = OrderedDict(
    [
        ("text-to-audio-spectrogram", TEXT_TO_AUDIO_SPECTROGRAM_SAMPLE),
        ("text-to-audio-waveform", TEXT_TO_AUDIO_WAVEFORM_SAMPLE),
        ("automatic-speech-recognition", AUTOMATIC_SPEECH_RECOGNITION_SAMPLE),
        ("audio-frame-classification", AUDIO_FRAME_CLASSIFICATION_SAMPLE),
        ("audio-classification", AUDIO_CLASSIFICATION_SAMPLE),
        ("audio-xvector", AUDIO_XVECTOR_SAMPLE),
        ("image-text-to-text", IMAGE_TEXT_TO_TEXT_GENERATION_SAMPLE),
        ("image-to-text", IMAGE_TO_TEXT_SAMPLE),
        ("visual-question-answering", VISUAL_QUESTION_ANSWERING_SAMPLE),
        ("depth-estimation", DEPTH_ESTIMATION_SAMPLE),
        ("video-classification", VIDEO_CLASSIFICATION_SAMPLE),
        ("zero-shot-image-classification", ZERO_SHOT_IMAGE_CLASSIFICATION_SAMPLE),
        ("image-classification", IMAGE_CLASSIFICATION_SAMPLE),
        ("zero-shot-object-detection", ZERO_SHOT_OBJECT_DETECTION_SAMPLE),
        ("object-detection", OBJECT_DETECTION_SAMPLE),
        ("image-segmentation", IMAGE_SEGMENTATION_SAMPLE),
        ("image-to-image", IMAGE_TO_IMAGE_SAMPLE),
        ("image-feature-extraction", IMAGE_FEATURE_EXTRACTION_SAMPLE),
        ("text-generation", TEXT_GENERATION_SAMPLE),
        ("table-question-answering", TABLE_QUESTION_ANSWERING_SAMPLE),
        ("document-question-answering", DOCUMENT_QUESTION_ANSWERING_SAMPLE),
        ("question-answering", QUESTION_ANSWERING_SAMPLE),
        ("text2text-generation", TEXT2TEXT_GENERATION_SAMPLE),
        ("next-sentence-prediction", NEXT_SENTENCE_PREDICTION_SAMPLE),
        ("multiple-choice", MULTIPLE_CHOICE_SAMPLE),
        ("text-classification", TEXT_CLASSIFICATION_SAMPLE),
        ("token-classification", TOKEN_CLASSIFICATION_SAMPLE),
        ("fill-mask", FILL_MASK_SAMPLE),
        ("mask-generation", MASK_GENERATION_SAMPLE),
        ("pretraining", PRETRAINING_SAMPLE),
    ]
)

# Ordered dict to look for more specialized model mappings first
# before falling back to the more generic ones.
MODELS_TO_PIPELINE = OrderedDict(
    [
        # Audio
        ("MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES", "text-to-audio-spectrogram"),
        ("MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES", "text-to-audio-waveform"),
        ("MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES", "automatic-speech-recognition"),
        ("MODEL_FOR_CTC_MAPPING_NAMES", "automatic-speech-recognition"),
        ("MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES", "audio-frame-classification"),
        ("MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES", "audio-classification"),
        ("MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES", "audio-xvector"),
        # Vision
        ("MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES", "image-text-to-text"),
        ("MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES", "image-to-text"),
        ("MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES", "visual-question-answering"),
        ("MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES", "depth-estimation"),
        ("MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES", "video-classification"),
        ("MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES", "zero-shot-image-classification"),
        ("MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES", "image-classification"),
        ("MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES", "zero-shot-object-detection"),
        ("MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES", "object-detection"),
        ("MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES", "image-segmentation"),
        ("MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES", "image-to-image"),
        ("MODEL_FOR_IMAGE_MAPPING_NAMES", "image-feature-extraction"),
        # Text/tokens
        ("MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", "text-generation"),
        ("MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES", "table-question-answering"),
        ("MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES", "document-question-answering"),
        ("MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES", "question-answering"),
        ("MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES", "text2text-generation"),
        ("MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES", "next-sentence-prediction"),
        ("MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES", "multiple-choice"),
        ("MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES", "text-classification"),
        ("MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES", "token-classification"),
        ("MODEL_FOR_MASKED_LM_MAPPING_NAMES", "fill-mask"),
        ("MODEL_FOR_MASK_GENERATION_MAPPING_NAMES", "mask-generation"),
        ("MODEL_FOR_PRETRAINING_MAPPING_NAMES", "pretraining"),
    ]
)


TF_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer(
    ...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="tf"
    ... )

    >>> logits = model(**inputs).logits
    >>> predicted_token_class_ids = tf.math.argmax(logits, axis=-1)

    >>> # Note that tokens are classified rather then input words which means that
    >>> # there might be more predicted token classes than words.
    >>> # Multiple token classes might account for the same word
    >>> predicted_tokens_classes = [model.config.id2label[t] for t in predicted_token_class_ids[0].numpy().tolist()]
    >>> predicted_tokens_classes
    {expected_output}
    ```

    ```python
    >>> labels = predicted_token_class_ids
    >>> loss = tf.math.reduce_mean(model(**inputs, labels=labels).loss)
    >>> round(float(loss), 2)
    {expected_loss}
    ```
"""

TF_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    >>> inputs = tokenizer(question, text, return_tensors="tf")
    >>> outputs = model(**inputs)

    >>> answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
    >>> answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

    >>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    >>> tokenizer.decode(predict_answer_tokens)
    {expected_output}
    ```

    ```python
    >>> # target is "nice puppet"
    >>> target_start_index = tf.constant([{qa_target_start_index}])
    >>> target_end_index = tf.constant([{qa_target_end_index}])

    >>> outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
    >>> loss = tf.math.reduce_mean(outputs.loss)
    >>> round(float(loss), 2)
    {expected_loss}
    ```
"""

TF_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")

    >>> logits = model(**inputs).logits

    >>> predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
    >>> model.config.id2label[predicted_class_id]
    {expected_output}
    ```

    ```python
    >>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
    >>> num_labels = len(model.config.id2label)
    >>> model = {model_class}.from_pretrained("{checkpoint}", num_labels=num_labels)

    >>> labels = tf.constant(1)
    >>> loss = model(**inputs, labels=labels).loss
    >>> round(float(loss), 2)
    {expected_loss}
    ```
"""

TF_MASKED_LM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="tf")
    >>> logits = model(**inputs).logits

    >>> # retrieve index of {mask}
    >>> mask_token_index = tf.where((inputs.input_ids == tokenizer.mask_token_id)[0])
    >>> selected_logits = tf.gather_nd(logits[0], indices=mask_token_index)

    >>> predicted_token_id = tf.math.argmax(selected_logits, axis=-1)
    >>> tokenizer.decode(predicted_token_id)
    {expected_output}
    ```

    ```python
    >>> labels = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]
    >>> # mask labels of non-{mask} tokens
    >>> labels = tf.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

    >>> outputs = model(**inputs, labels=labels)
    >>> round(float(outputs.loss), 2)
    {expected_loss}
    ```
"""

TF_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    >>> outputs = model(inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

TF_MULTIPLE_CHOICE_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    >>> choice0 = "It is eaten with a fork and a knife."
    >>> choice1 = "It is eaten while held in the hand."

    >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="tf", padding=True)
    >>> inputs = {{k: tf.expand_dims(v, 0) for k, v in encoding.items()}}
    >>> outputs = model(inputs)  # batch size is 1

    >>> # the linear classifier still needs to be trained
    >>> logits = outputs.logits
    ```
"""

TF_CAUSAL_LM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}
    >>> import tensorflow as tf

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    >>> outputs = model(inputs)
    >>> logits = outputs.logits
    ```
"""

TF_SPEECH_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoProcessor, {model_class}
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="tf")
    >>> outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    >>> list(last_hidden_states.shape)
    {expected_output}
    ```
"""

TF_SPEECH_CTC_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoProcessor, {model_class}
    >>> from datasets import load_dataset
    >>> import tensorflow as tf

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = AutoProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="tf")
    >>> logits = model(**inputs).logits
    >>> predicted_ids = tf.math.argmax(logits, axis=-1)

    >>> # transcribe speech
    >>> transcription = processor.batch_decode(predicted_ids)
    >>> transcription[0]
    {expected_output}
    ```

    ```python
    >>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="tf").input_ids

    >>> # compute loss
    >>> loss = model(**inputs).loss
    >>> round(float(loss), 2)
    {expected_loss}
    ```
"""

TF_VISION_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoImageProcessor, {model_class}
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("huggingface/cats-image")
    >>> image = dataset["test"]["image"][0]

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = image_processor(image, return_tensors="tf")
    >>> outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    >>> list(last_hidden_states.shape)
    {expected_output}
    ```
"""

TF_VISION_SEQ_CLASS_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoImageProcessor, {model_class}
    >>> import tensorflow as tf
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("huggingface/cats-image"))
    >>> image = dataset["test"]["image"][0]

    >>> image_processor = AutoImageProcessor.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = image_processor(image, return_tensors="tf")
    >>> logits = model(**inputs).logits

    >>> # model predicts one of the 1000 ImageNet classes
    >>> predicted_label = int(tf.math.argmax(logits, axis=-1))
    >>> print(model.config.id2label[predicted_label])
    {expected_output}
    ```
"""

TF_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": TF_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": TF_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": TF_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": TF_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": TF_MASKED_LM_SAMPLE,
    "LMHead": TF_CAUSAL_LM_SAMPLE,
    "BaseModel": TF_BASE_MODEL_SAMPLE,
    "SpeechBaseModel": TF_SPEECH_BASE_MODEL_SAMPLE,
    "CTC": TF_SPEECH_CTC_SAMPLE,
    "VisionBaseModel": TF_VISION_BASE_MODEL_SAMPLE,
    "ImageClassification": TF_VISION_SEQ_CLASS_SAMPLE,
}


FLAX_TOKEN_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""

FLAX_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    >>> inputs = tokenizer(question, text, return_tensors="jax")

    >>> outputs = model(**inputs)
    >>> start_scores = outputs.start_logits
    >>> end_scores = outputs.end_logits
    ```
"""

FLAX_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""

FLAX_MASKED_LM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="jax")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""

FLAX_BASE_MODEL_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
    >>> outputs = model(**inputs)

    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""

FLAX_MULTIPLE_CHOICE_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    >>> choice0 = "It is eaten with a fork and a knife."
    >>> choice1 = "It is eaten while held in the hand."

    >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="jax", padding=True)
    >>> outputs = model(**{{k: v[None, :] for k, v in encoding.items()}})

    >>> logits = outputs.logits
    ```
"""

FLAX_CAUSAL_LM_SAMPLE = r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, {model_class}

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
    >>> outputs = model(**inputs)

    >>> # retrieve logts for next token
    >>> next_token_logits = outputs.logits[:, -1]
    ```
"""

FLAX_SAMPLE_DOCSTRINGS = {
    "SequenceClassification": FLAX_SEQUENCE_CLASSIFICATION_SAMPLE,
    "QuestionAnswering": FLAX_QUESTION_ANSWERING_SAMPLE,
    "TokenClassification": FLAX_TOKEN_CLASSIFICATION_SAMPLE,
    "MultipleChoice": FLAX_MULTIPLE_CHOICE_SAMPLE,
    "MaskedLM": FLAX_MASKED_LM_SAMPLE,
    "BaseModel": FLAX_BASE_MODEL_SAMPLE,
    "LMHead": FLAX_CAUSAL_LM_SAMPLE,
}


def filter_outputs_from_example(docstring, **kwargs):
    """
    Removes the lines testing an output with the doctest syntax in a code sample when it's set to `None`.
    """
    for key, value in kwargs.items():
        if value is not None:
            continue

        doc_key = "{" + key + "}"
        docstring = re.sub(rf"\n([^\n]+)\n\s+{doc_key}\n", "\n", docstring)

    return docstring


def add_code_sample_docstrings(
    *docstr,
    processor_class=None,
    checkpoint=None,
    output_type=None,
    config_class=None,
    mask="[MASK]",
    qa_target_start_index=14,
    qa_target_end_index=15,
    model_cls=None,
    modality=None,
    expected_output=None,
    expected_loss=None,
    real_checkpoint=None,
    revision=None,
):
    def docstring_decorator(fn):
        # model_class defaults to function's class if not specified otherwise
        model_class = fn.__qualname__.split(".")[0] if model_cls is None else model_cls

        if model_class[:2] == "TF":
            sample_docstrings = TF_SAMPLE_DOCSTRINGS
        elif model_class[:4] == "Flax":
            sample_docstrings = FLAX_SAMPLE_DOCSTRINGS
        else:
            sample_docstrings = PT_SAMPLE_DOCSTRINGS

        # putting all kwargs for docstrings in a dict to be used
        # with the `.format(**doc_kwargs)`. Note that string might
        # be formatted with non-existing keys, which is fine.
        doc_kwargs = {
            "model_class": model_class,
            "processor_class": processor_class,
            "checkpoint": checkpoint,
            "mask": mask,
            "qa_target_start_index": qa_target_start_index,
            "qa_target_end_index": qa_target_end_index,
            "expected_output": expected_output,
            "expected_loss": expected_loss,
            "real_checkpoint": real_checkpoint,
            "fake_checkpoint": checkpoint,
            "true": "{true}",  # For <Tip warning={true}> syntax that conflicts with formatting.
        }

        if ("SequenceClassification" in model_class or "AudioClassification" in model_class) and modality == "audio":
            code_sample = sample_docstrings["AudioClassification"]
        elif "SequenceClassification" in model_class:
            code_sample = sample_docstrings["SequenceClassification"]
        elif "QuestionAnswering" in model_class:
            code_sample = sample_docstrings["QuestionAnswering"]
        elif "TokenClassification" in model_class:
            code_sample = sample_docstrings["TokenClassification"]
        elif "MultipleChoice" in model_class:
            code_sample = sample_docstrings["MultipleChoice"]
        elif "MaskedLM" in model_class or model_class in ["FlaubertWithLMHeadModel", "XLMWithLMHeadModel"]:
            code_sample = sample_docstrings["MaskedLM"]
        elif "LMHead" in model_class or "CausalLM" in model_class:
            code_sample = sample_docstrings["LMHead"]
        elif "CTC" in model_class:
            code_sample = sample_docstrings["CTC"]
        elif "AudioFrameClassification" in model_class:
            code_sample = sample_docstrings["AudioFrameClassification"]
        elif "XVector" in model_class and modality == "audio":
            code_sample = sample_docstrings["AudioXVector"]
        elif "Model" in model_class and modality == "audio":
            code_sample = sample_docstrings["SpeechBaseModel"]
        elif "Model" in model_class and modality == "vision":
            code_sample = sample_docstrings["VisionBaseModel"]
        elif "Model" in model_class or "Encoder" in model_class:
            code_sample = sample_docstrings["BaseModel"]
        elif "ImageClassification" in model_class:
            code_sample = sample_docstrings["ImageClassification"]
        else:
            raise ValueError(f"Docstring can't be built for model {model_class}")

        code_sample = filter_outputs_from_example(
            code_sample, expected_output=expected_output, expected_loss=expected_loss
        )
        if real_checkpoint is not None:
            code_sample = FAKE_MODEL_DISCLAIMER + code_sample
        func_doc = (fn.__doc__ or "") + "".join(docstr)
        output_doc = "" if output_type is None else _prepare_output_docstrings(output_type, config_class)
        built_doc = code_sample.format(**doc_kwargs)
        if revision is not None:
            if re.match(r"^refs/pr/\\d+", revision):
                raise ValueError(
                    f"The provided revision '{revision}' is incorrect. It should point to"
                    " a pull request reference on the hub like 'refs/pr/6'"
                )
            built_doc = built_doc.replace(
                f'from_pretrained("{checkpoint}")', f'from_pretrained("{checkpoint}", revision="{revision}")'
            )

        fn.__doc__ = func_doc + output_doc + built_doc
        return fn

    return docstring_decorator


def replace_return_docstrings(output_type=None, config_class=None):
    def docstring_decorator(fn):
        func_doc = fn.__doc__
        lines = func_doc.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^\s*Returns?:\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = len(_get_indent(lines[i]))
            lines[i] = _prepare_output_docstrings(output_type, config_class, min_indent=indent)
            func_doc = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, "
                f"current docstring is:\n{func_doc}"
            )
        fn.__doc__ = func_doc
        return fn

    return docstring_decorator


def copy_func(f):
    """Returns a copy of a function f."""
    # Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g
