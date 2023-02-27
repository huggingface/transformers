from typing import Union

import numpy as np
import requests

from ..utils import (
    add_end_docstrings,
    is_torch_available,
    logging,
)
from .audio_classification import ffmpeg_read
from .base import PIPELINE_INIT_ARGS, ChunkPipeline


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class ZeroShotAudioClassificationPipeline(ChunkPipeline):
    """
    Zero shot audio classification pipeline using `ClapModel`. This pipeline predicts the class of an audio when you
    provide an audio and a set of `candidate_labels`.

    Example:
    ```python
    >>> from transformers import pipeline
    >>> from datasets import load_dataset

    >>> dataset = load_dataset("ashraq/esc50")
    >>> audio = next(iter(dataset["train"]["audio"]))["array"]
    >>> classifier = pipeline(task="zero-shot-audio-classification", model="laion-ai/clap-htsat-tiny")
    >>> classifier(audio, candidate_labels=["Sound of a dog", "Sound of vaccum cleaner"])
    ```


    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial) This audio
    classification pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"zero-shot-audio-classification"`. See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=zero-shot-audio-classification).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.framework != "pt":
            raise ValueError(f"The {self.__class__} is only available in PyTorch.")
        # No specific FOR_XXX available yet

    def __call__(self, audios: Union[np.ndarray, bytes, str], **kwargs):
        """
        Assign labels to the audio(s) passed as inputs.

        Args:
            audios (`str`, `List[str]`, `np.array` or `List[np.array]`):
                The pipeline handles three types of inputs:
                - A string containing a http link pointing to an audio
                - A string containing a local path to an audio
                - An audio loaded in numpy
            candidate_labels (`List[str]`):
                The candidate labels for this audio
            hypothesis_template (`str`, *optional*, defaults to `"This is a sound of {}"`):
                The sentence used in cunjunction with *candidate_labels* to attempt the audio classification by
                replacing the placeholder with the candidate_labels. Then likelihood is estimated by using
                logits_per_audio
        Return:
            A list of dictionaries containing result, one dictionary per proposed label. The dictionaries contain the
            following keys:
            - **label** (`str`) -- The label identified by the model. It is one of the suggested `candidate_label`.
            - **score** (`float`) -- The score attributed by the model for that label (between 0 and 1).
        """
        return super().__call__(audios, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        if "candidate_labels" in kwargs:
            preprocess_params["candidate_labels"] = kwargs["candidate_labels"]
        if "hypothesis_template" in kwargs:
            preprocess_params["hypothesis_template"] = kwargs["hypothesis_template"]

        return preprocess_params, {}, {}

    def preprocess(self, audio, candidate_labels=None, hypothesis_template="This is a sound of {}."):
        if isinstance(audio, str):
            if audio.startswith("http://") or audio.startswith("https://"):
                # We need to actually check for a real protocol, otherwise it's impossible to use a local file
                # like http_huggingface_co.png
                audio = requests.get(audio).content
            else:
                with open(audio, "rb") as f:
                    audio = f.read()

        if isinstance(audio, bytes):
            audio = ffmpeg_read(audio, self.feature_extractor.sampling_rate)

        if not isinstance(audio, np.ndarray):
            raise ValueError("We expect a numpy ndarray as input")
        if len(audio.shape) != 1:
            raise ValueError("We expect a single channel audio input for ZeroShotAudioClassificationPipeline")

        n = len(candidate_labels)
        for i, candidate_label in enumerate(candidate_labels):
            audios = self.feature_extractor(
                audio, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
            )
            sequence = hypothesis_template.format(candidate_label)
            inputs = self.tokenizer(sequence, return_tensors=self.framework)
            inputs["input_features"] = audios.input_features
            yield {"is_last": i == n - 1, "candidate_label": candidate_label, **inputs}

    def _forward(self, model_inputs):
        is_last = model_inputs.pop("is_last")
        candidate_label = model_inputs.pop("candidate_label")
        outputs = self.model(**model_inputs)

        # Clap does crossproduct scoring by default, so we're only
        # interested in the results where audio and text and in the same
        # batch position.
        diag = torch.diagonal
        logits_per_audio = diag(outputs.logits_per_audio)

        model_outputs = {
            "is_last": is_last,
            "candidate_label": candidate_label,
            "logits_per_audio": logits_per_audio,
        }
        return model_outputs

    def postprocess(self, model_outputs):
        candidate_labels = [outputs["candidate_label"] for outputs in model_outputs]
        if self.framework == "pt":
            logits = torch.cat([output["logits_per_audio"] for output in model_outputs])
            probs = logits.softmax(dim=0)
            scores = probs.tolist()
        else:
            raise ValueError("`tf` framework not supported.")

        result = [
            {"score": score, "label": candidate_label}
            for score, candidate_label in sorted(zip(scores, candidate_labels), key=lambda x: -x[0])
        ]
        return result
