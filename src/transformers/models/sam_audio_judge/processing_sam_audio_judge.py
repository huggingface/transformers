from typing import Optional, Union

from ...audio_utils import AudioInput
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import (
    PreTokenizedInput,
    TextInput,
)


class SamAudioJudgeProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "PeAudioFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]]] = None,
        input_audio: Optional[AudioInput] = None,
        separated_audio: Optional[AudioInput] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        res = super().__call__(text=text, audio=input_audio, **kwargs).data
        if separated_audio is not None:
            kwargs = self._merge_kwargs(
                self.valid_processor_kwargs,
                tokenizer_init_kwargs=self.tokenizer.init_kwargs if hasattr(self, "tokenizer") else {},
                **kwargs,
            )

            res["separated_values"] = self.feature_extractor(separated_audio, **kwargs["audio_kwargs"])["input_values"]
        return BatchFeature(res)


__all__ = ["SamAudioJudgeProcessor"]
