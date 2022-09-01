from typing import List, Union

from ..utils import (
    add_end_docstrings,
    is_tf_available,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_tf_available():
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_VISION_2_SEQ_MAPPING

if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING

logger = logging.get_logger(__name__)


@add_end_docstrings(PIPELINE_INIT_ARGS)
class Image2TextGenerationPipeline(Pipeline):
    """
    Image2Text Generation pipeline using a `AutoModelForVision2Seq`. This pipeline predicts a caption for a given
    image.

    This image to text generation pipeline can currently be loaded from pipeline() using the following task identifier:
    "image2text-generation".

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?pipeline_tag=image-to-text).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "vision")
        self.check_model_type(
            TF_MODEL_FOR_VISION_2_SEQ_MAPPING if self.framework == "tf" else MODEL_FOR_VISION_2_SEQ_MAPPING
        )

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def __call__(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], **kwargs):
        """
        Assign labels to the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a HTTP(s) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images.

        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following key:

            - **generated_text** (`str`) -- The generated text.
        """
        return super().__call__(images, **kwargs)

    def preprocess(self, image):
        image = load_image(image)
        model_inputs = self.feature_extractor(images=image, return_tensors=self.framework)
        return model_inputs

    def _forward(self, model_inputs):
        # FIXME: We need to pop here due to a difference in how `generation_utils.py` and `generation_tf_utils.py`
        #  parse inputs. In the Tensorflow version, `generate` raises an error if we don't use `input_ids` whereas
        #  the PyTorch version matches it with `self.model.main_input_name` or `self.model.encoder.main_input_name`
        #  in the `_prepare_model_inputs` method.
        inputs = model_inputs.pop(self.model.main_input_name)
        model_outputs = self.model.generate(inputs, **model_inputs)
        return model_outputs

    def postprocess(self, model_outputs):
        records = []
        for output_ids in model_outputs:
            record = {
                "generated_text": self.tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                )
            }
            records.append(record)
        return records
