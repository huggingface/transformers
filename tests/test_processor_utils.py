import tempfile

from transformers.testing_utils import TestCasePlus
from transformers import ProcessorMixin, AutoTokenizer, PreTrainedTokenizer


class ProcessorSavePretrainedMultipleAttributes(TestCasePlus):
    def test_processor_loads_separate_attributes(self):
        class OtherProcessor(ProcessorMixin):
            name = "other-processor"

            attributes = [
                "tokenizer1",
                "tokenizer2",
            ]
            tokenizer1_class = "AutoTokenizer"
            tokenizer2_class = "AutoTokenizer"

            def __init__(self,
                         tokenizer1: PreTrainedTokenizer,
                         tokenizer2: PreTrainedTokenizer
                         ):
                super().__init__(tokenizer1=tokenizer1,
                                 tokenizer2=tokenizer2)

        tokenizer1 = AutoTokenizer.from_pretrained("google/gemma-3-270m")
        tokenizer2 = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")

        processor = OtherProcessor(tokenizer1=tokenizer1,
                                   tokenizer2=tokenizer2)
        assert processor.tokenizer1.__class__ != processor.tokenizer2.__class__

        with tempfile.TemporaryDirectory() as temp_dir:
            processor.save_pretrained(save_directory=temp_dir, push_to_hub=False)
            new_processor = OtherProcessor.from_pretrained(temp_dir)

        assert new_processor.tokenizer1.__class__ != new_processor.tokenizer2.__class__
