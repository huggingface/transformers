from datasets import load_dataset

from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .base import PipelineTool


class SpeechToTextTool(PipelineTool):
    pre_processor_class = WhisperProcessor
    model_class = WhisperForConditionalGeneration

    description = """
    speech to text tool, used to convert a sound into text. It takes sound as input, and outputs a text.
    """

    def encode(self, inputs):
        return self.pre_processor(inputs, return_tensors="pt").input_features

    def forward(self, inputs):
        return self.model.generate(inputs=inputs)

    def decode(self, outputs):
        return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0]

