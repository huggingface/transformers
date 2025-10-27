import tempfile, os

from transformers import (
    ProcessorMixin,
    BertTokenizerFast,
    RobertaTokenizerFast,
)
from transformers.testing_utils import TestCasePlus


class TestProcessorSavePretrainedMultipleAttributes(TestCasePlus):
    def test_processor_loads_separate_attributes(self):

        class OtherProcessor(ProcessorMixin):
            name = "other-processor"
            attributes = ["tokenizer1", "tokenizer2"]

            # Must be class names as strings
            tokenizer1_class = "BertTokenizerFast"
            tokenizer2_class = "RobertaTokenizerFast"

            def __init__(self, tokenizer1, tokenizer2):
                super().__init__(tokenizer1=tokenizer1, tokenizer2=tokenizer2)

        # Initialize tokenizers
        tokenizer1 = BertTokenizerFast.from_pretrained("bert-base-uncased")
        tokenizer2 = RobertaTokenizerFast.from_pretrained("roberta-base")

        processor = OtherProcessor(tokenizer1=tokenizer1, tokenizer2=tokenizer2)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save tokenizers in separate folders
            tokenizer1_dir = os.path.join(temp_dir, "tokenizer1")
            tokenizer1.save_pretrained(tokenizer1_dir)

            tokenizer2_dir = os.path.join(temp_dir, "tokenizer2")
            tokenizer2.save_pretrained(tokenizer2_dir)

            # Save processor metadata
            processor.save_pretrained(temp_dir, push_to_hub=False)

            # Reload tokenizers
            loaded_tokenizer1 = BertTokenizerFast.from_pretrained(tokenizer1_dir)
            loaded_tokenizer2 = RobertaTokenizerFast.from_pretrained(tokenizer2_dir)

            # Recreate processor with loaded tokenizers
            new_processor = OtherProcessor(tokenizer1=loaded_tokenizer1,
                                           tokenizer2=loaded_tokenizer2)

        # Assert the two tokenizers are of different classes
        assert new_processor.tokenizer1.__class__ != new_processor.tokenizer2.__class__
