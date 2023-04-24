import torch
from datasets import load_dataset

from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from .base import PipelineTool


class TextToSpeechTool(PipelineTool):
    pre_processor_class = SpeechT5Processor
    model_class = SpeechT5ForTextToSpeech
    post_processor_class = SpeechT5HifiGan

    description = (
        "This is a tool that reads an English text out loud. It takes an input named `text` which whould contain the "
        "text to read (in English) and returns a waveform object containing the sound."
    )

    def __init__(self, model=None, pre_processor=None, post_processor=None, **kwargs):
        if model is None and pre_processor is None and post_processor is None:
            model = "microsoft/speecht5_tts"
            pre_processor = "microsoft/speecht5_tts"
            post_processor = "microsoft/speecht5_hifigan"

        super().__init__(model, pre_processor, post_processor, **kwargs)

    def encode(self, text, speaker_embeddings=None):
        inputs = self.pre_processor(text=text, return_tensors="pt")

        if speaker_embeddings is None:
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[7305]["xvector"]).unsqueeze(0)

        return {"input_ids": inputs["input_ids"], "speaker_embeddings": speaker_embeddings}

    def forward(self, inputs):
        with torch.no_grad():
            return self.model.generate_speech(**inputs)

    def decode(self, outputs):
        with torch.no_grad():
            return self.post_processor(outputs).cpu().detach()
