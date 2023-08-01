from typing import List, Union

import numpy as np

from transformers import Pipeline, SpeechT5HifiGan

from ..utils import is_torch_available


if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING

DEFAULT_VOCODER_ID = "microsoft/speecht5_hifigan"


class TextToAudioPipeline(Pipeline):
    """
    Text-to-audio generation pipeline using any `AutoModelForTextToWaveform` or `AutoModelForTextToSpectrogram`. This
    pipeline generates an audio file from an input text and optional other conditional inputs.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> classifier = pipeline(model="suno/bark")
    >>> audio = pipeline("Hey it's HuggingFace on the phone!", speaker_embeddings="v2/en_speaker_1")
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)


    This pipeline can currently be loaded from [`pipeline`] using the following task identifiers: `"text-to-speech"` or
    `"text-to-audio"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=text-to-speech).
    """

    def __init__(self, *args, vocoder=None, sampling_rate=None, **kwargs):
        super().__init__(*args, **kwargs)

        if self.framework == "tf":
            raise ValueError("The TextToAudioPipeline is only available in PyTorch.")

        self.forward_method = self.model.generate if self.model.can_generate() else self.model
        
        self.vocoder = None
        if self.model.__class__ in MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING.values():
            self.vocoder = (
                SpeechT5HifiGan.from_pretrained(DEFAULT_VOCODER_ID).to(self.model.device)
                if vocoder is None
                else vocoder
            )

        self.sampling_rate = sampling_rate
        if self.vocoder is not None:
            self.sampling_rate = self.vocoder.config.sampling_rate

        if self.sampling_rate is None:
            # get sampling_rate from config and generation config
            self.sampling_rate = None

            config = self.model.config.to_dict()
            gen_config = self.model.__dict__.get("generation_config", None)
            if gen_config is not None:
                config.update(gen_config.to_dict())

            for sampling_rate_name in ["sample_rate", "sampling_rate"]:
                sampling_rate = config.get(sampling_rate_name, None)
                if sampling_rate is not None:
                    self.sampling_rate = sampling_rate

    def preprocess(self, text, **kwargs):
        if isinstance(text, str):
            text = [text]

        output = self.tokenizer(text, **kwargs, return_tensors="pt")

        return output

    def _forward(self, model_inputs, **kwargs):
        
        # we expect some kwargs to be additional tensors which need to be on the right device 
        kwargs = self._ensure_tensor_on_device(kwargs, device=self.device)

        # call the generate by defaults or the forward method if the model cannot generate
        output = self.forward_method(**model_inputs, **kwargs)

        if self.vocoder is not None:
            # in that case, the output is a spectrogram that needs to be converted into a waveform
            output = self.vocoder(output)

        return output

    def __call__(
        self,
        input_texts: Union[str, List[str]],
        **forward_params,
    ):
        """
        Generates speech/audio from the inputs. See the [`TextToAudioPipeline`] documentation for more information.

        Args:
            input_texts (`str` or `List[str]`):
                The text(s) to generate.
            forward_params (*optional*):
                Parameters passed to the model generation/forward method.

        Return:
            A `torch.Tensor` or a list of `torch.Tensor`: Each result comes as a `torch.Tensor` corresponding to the
            generated audio.
        """
        return super().__call__(input_texts, **forward_params)

    def _sanitize_parameters(
        self,
        preprocess_params=None,
        forward_params=None,
    ):
        if preprocess_params is None:
            preprocess_params = {}
        if forward_params is None:
            forward_params = {}
        postprocess_params = {}

        return preprocess_params, forward_params, postprocess_params

    def postprocess(self, waveform):
        output_dict = {}

        output_dict["audio"] = waveform.detach().cpu().numpy().squeeze().astype(np.float32)
        output_dict["sampling_rate"] = self.sampling_rate

        return output_dict
