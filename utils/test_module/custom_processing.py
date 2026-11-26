from transformers import ProcessorMixin


class CustomProcessor(ProcessorMixin):
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)


class CustomMultimodalProcessor(ProcessorMixin):
    def __init__(self, image_processor, video_processor, feature_extractor, tokenizer):
        super().__init__(image_processor, video_processor, feature_extractor, tokenizer)
