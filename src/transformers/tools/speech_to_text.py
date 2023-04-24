from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .base import PipelineTool


class SpeechToTextTool(PipelineTool):
    pre_processor_class = WhisperProcessor
    model_class = WhisperForConditionalGeneration

    description = (
        "This is a tool that transcribes an audio into text. It takes an input named `audio` and returns the "
        "transcribed text."
    )

    def encode(self, audio):
        return self.pre_processor(audio, return_tensors="pt").input_features

    def forward(self, inputs):
        return self.model.generate(inputs=inputs)

    def decode(self, outputs):
        return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0]
