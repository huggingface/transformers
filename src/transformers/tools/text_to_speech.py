import torch

from transformers.utils import is_datasets_available

from ..models.speecht5 import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
from .base import PipelineTool


if is_datasets_available():
    from datasets import load_dataset


TEXT_TO_SPEECH_DESCRIPTION = (
    "This is a tool that reads an English text out loud. It takes an input named `text` which whould contain the "
    "text to read (in English) and returns a waveform object containing the sound."
)


class TextToSpeechTool(PipelineTool):
    default_checkpoint = "microsoft/speecht5_tts"
    description = TEXT_TO_SPEECH_DESCRIPTION
    pre_processor_class = SpeechT5Processor
    model_class = SpeechT5ForTextToSpeech
    post_processor_class = SpeechT5HifiGan

    def post_init(self):
        if self.post_processor is None:
            self.post_processor = "microsoft/speecht5_hifigan"

    def encode(self, text, speaker_embeddings=None):
        inputs = self.pre_processor(text=text, return_tensors="pt")

        if speaker_embeddings is None:
            if not is_datasets_available():
                raise ImportError("Datasets needs to be installed if not passing speaker embeddings.")

            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[7305]["xvector"]).unsqueeze(0)

        return {"input_ids": inputs["input_ids"], "speaker_embeddings": speaker_embeddings}

    def forward(self, inputs):
        with torch.no_grad():
            return self.model.generate_speech(**inputs)

    def decode(self, outputs):
        with torch.no_grad():
            return self.post_processor(outputs).cpu().detach()
