import re

from transformers import VisionEncoderDecoderModel

from ..models.auto import AutoProcessor
from ..utils import is_vision_available
from .base import PipelineTool


if is_vision_available():
    from PIL import Image


DOCUMENT_QUESTION_ANSWERING_DESCRIPTION = (
    "This is a tool that answers a question about an document (pdf). It takes an input named `document` which should be the "
    "document containing the information, as well as a `query` that is the question about the document. It returns a text "
    "that contains the answer to the question."
)


class DocumentQuestionAnsweringTool(PipelineTool):
    default_checkpoint = "naver-clova-ix/donut-base-finetuned-docvqa"
    description = DOCUMENT_QUESTION_ANSWERING_DESCRIPTION
    pre_processor_class = AutoProcessor
    model_class = VisionEncoderDecoderModel

    inputs = ["image", "text"]
    outputs = ["text"]

    def __init__(self, *args, **kwargs):
        if not is_vision_available():
            raise ValueError("Pillow must be installed to use the DocumentQuestionAnsweringTool.")

        super().__init__(*args, **kwargs)

    def encode(self, image: "Image", query: str):
        task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
        prompt = task_prompt.replace("{user_input}", query)
        decoder_input_ids = self.pre_processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        pixel_values = self.pre_processor(image, return_tensors="pt").pixel_values

        return {'decoder_input_ids': decoder_input_ids, 'pixel_values': pixel_values}

    def forward(self, inputs):
        return self.model.generate(
            inputs['pixel_values'].to(self.device),
            decoder_input_ids=inputs['decoder_input_ids'].to(self.device),
            max_length=self.model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=self.pre_processor.tokenizer.pad_token_id,
            eos_token_id=self.pre_processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.pre_processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        ).sequences

    def decode(self, outputs):
        sequence = self.pre_processor.batch_decode(outputs)[0]
        sequence = sequence.replace(self.pre_processor.tokenizer.eos_token, "").replace(self.pre_processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
        sequence = self.pre_processor.token2json(sequence)

        return sequence['answer']

