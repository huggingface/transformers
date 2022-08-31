import enum
from typing import List, Union

from ..utils import (
    add_end_docstrings,
    is_flax_available,
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

if is_flax_available():
    from ..models.auto.modeling_flax_auto import FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING

if is_tf_available():
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_VISION_2_SEQ_MAPPING

if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING

logger = logging.get_logger(__name__)


class ReturnType(enum.Enum):
    TENSORS = 0
    TEXT = 1


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
        if self.framework == "flax":
            self.check_model_type(FLAX_MODEL_FOR_VISION_2_SEQ_MAPPING)
        elif self.framework == "tf":
            self.check_model_type(TF_MODEL_FOR_VISION_2_SEQ_MAPPING)
        else:
            self.check_model_type(MODEL_FOR_VISION_2_SEQ_MAPPING)

    def _sanitize_parameters(
        self, return_tensors=None, return_text=None, return_type=None, clean_up_tokenization_spaces=None, **kwargs
    ):
        preprocess_params, forward_params, postprocess_params = {}, {}, {}

        if return_tensors is not None and return_type is None:
            return_type = ReturnType.TENSORS if return_tensors else ReturnType.TEXT
        if return_type is not None:
            postprocess_params["return_type"] = return_type

        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

        return preprocess_params, forward_params, postprocess_params

    def __call__(self, images: Union[str, List[str], "Image.Image", List["Image.Image"]], **kwargs):
        """
        Assign labels to the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
                images.

        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following keys:

            - **generated_text** (`str`, present when `return_text=True`) -- The generated text.
            - **generated_token_ids** (`torch.Tensor`, present when `return_tensors=True`) -- The token ids of the
              generated text.
        """
        return super().__call__(images, **kwargs)

    def preprocess(self, image):
        image = load_image(image)
        model_inputs = self.feature_extractor(images=image, return_tensors=self.framework)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model.generate(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, return_type=ReturnType.TEXT, clean_up_tokenization_spaces=False):
        records = []
        for output_ids in model_outputs:
            if return_type == ReturnType.TENSORS:
                record = {"generated_token_ids": output_ids}
            elif return_type == ReturnType.TEXT:
                record = {
                    "generated_text": self.tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                }
            records.append(record)
        return records
