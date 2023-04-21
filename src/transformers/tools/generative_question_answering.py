import torch
from datasets import load_dataset

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base import PipelineTool


class GenerativeQuestionAnsweringTool(PipelineTool):
    pre_processor_class = AutoTokenizer
    model_class = AutoModelForSeq2SeqLM

    description = """
    Question answering tool, which can understand a large text and return the answer to a question.
    It takes the text as a first parameter, followed by the question.
    """
    default_checkpoint = 'google/flan-t5-base'

    def encode(self, text: str, question: str):
        prompt = f"""
        Here is a text containing a lot of information: '''{text}'''.

        Can you answer this question about the text: '{question}'
        """
        return self.pre_processor(prompt, return_tensors='pt')

    def forward(self, inputs):
        output_ids = self.model.generate(**inputs)

        in_b, _ = inputs["input_ids"].shape
        out_b = output_ids.shape[0]

        return output_ids.reshape(in_b, out_b // in_b, *output_ids.shape[1:])[0][0]

    def decode(self, outputs):
        return self.pre_processor.decode(
            outputs, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )