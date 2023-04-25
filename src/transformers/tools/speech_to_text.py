from ..models.whisper import WhisperForConditionalGeneration, WhisperProcessor
from .base import PipelineTool, RemoteTool


class SpeechToTextTool(PipelineTool):
    default_checkpoint = "openai/whisper-base"
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


class RemoteSpeechToTextTool(RemoteTool):
    default_checkpoint = "openai/whisper-base"
    description = (
        "This is a tool that transcribes an audio into text. It takes an input named `audio` and returns the "
        "transcribed text."
    )

    def extract_outputs(self, outputs):
        return outputs["text"]

    def prepare_inputs(self, audio):
        return {"data": audio}
