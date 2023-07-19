import torch
from datasets import load_dataset

from transformers import Pipeline, SpeechT5HifiGan, SpeechT5Processor


class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class


class TTSPipeline(Pipeline):
    def __init__(self, *args, vocoder=None, processor=None, **kwargs):
        super().__init__(*args, **kwargs)

        if vocoder is None:
            raise ValueError("Must pass a vocoder to the TTSPipeline.")

        if processor is None:
            raise ValueError("Must pass a processor to the TTSPipeline.")

        if isinstance(vocoder, str):
            vocoder = SpeechT5HifiGan.from_pretrained(vocoder)

        if isinstance(processor, str):
            processor = SpeechT5Processor.from_pretrained(processor)

        self.processor = processor
        self.vocoder = vocoder

    def preprocess(self, text, speaker_embeddings=None):
        inputs = self.processor(text=text, return_tensors="pt")

        if speaker_embeddings is None:
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[7305]["xvector"]).unsqueeze(0)

        return {"inputs": inputs, "speaker_embeddings": speaker_embeddings}

    def _forward(self, model_inputs):
        inputs = model_inputs["inputs"]
        speaker_embeddings = model_inputs["speaker_embeddings"]

        with torch.no_grad():
            speech = self.model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=self.vocoder)

        return speech

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, {}, {}

    def postprocess(self, speech):
        return speech
