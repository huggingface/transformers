import tempfile
import unittest

from transformers import ChameleonProcessor
from transformers.models.auto.processing_auto import processor_class_from_name

from ...test_processing_common import ProcessorTesterMixin


class ChameleonProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    from_pretrained_id = "leloy/Anole-7b-v0.1-hf"
    processor_class = ChameleonProcessor

    def get_component(self, attribute, **kwargs):
        assert attribute in self.processor_class.attributes
        component_class_name = getattr(self.processor_class, f"{attribute}_class")
        if isinstance(component_class_name, tuple):
            if "_fast" in component_class_name[0]:
                component_class_name = component_class_name[0]
            else:
                component_class_name = component_class_name[1]

        component_class = processor_class_from_name(component_class_name)
        component = component_class.from_pretrained(self.tmpdirname, **kwargs)  # noqa
        if attribute == "tokenizer" and not component.pad_token:
            component.pad_token = "[TEST_PAD]"

        return component

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()
        processor = self.processor_class.from_pretrained(self.from_pretrained_id)
        processor.save_pretrained(self.tmpdirname)
