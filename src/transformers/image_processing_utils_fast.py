from .image_processing_base import ImageProcessingMixin


class BaseImageProcessorFast(ImageProcessingMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, images, **kwargs):
        return self.preprocess(images, **kwargs)

    def preprocess(self, images, **kwargs):
        raise NotImplementedError

