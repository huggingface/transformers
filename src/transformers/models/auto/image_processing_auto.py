# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AutoImageProcessor class."""

import importlib
import os
from collections import OrderedDict
from typing import TYPE_CHECKING

# Build the list of all image processors
from ...configuration_utils import PreTrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...image_processing_utils import ImageProcessingMixin
from ...utils import (
    CONFIG_NAME,
    IMAGE_PROCESSOR_NAME,
    PROCESSOR_NAME,
    cached_file,
    is_timm_config_dict,
    is_timm_local_checkpoint,
    is_torchvision_available,
    logging,
    safe_load_json_file,
)
from ...utils.import_utils import requires
from .auto_factory import _LazyAutoMapping
from .auto_mappings import IMAGE_PROCESSOR_MAPPING_NAMES
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)


logger = logging.get_logger(__name__)

# These image processors use Lanczos interpolation, which is not supported by fast image processors.
# To avoid important differences in outputs, we default to using the PIL backend for these processors.
DEFAULT_TO_PIL_BACKEND_IMAGE_PROCESSORS = [
    "ChameleonImageProcessor",
    "FlavaImageProcessor",
    "Idefics3ImageProcessor",
    "SmolVLMImageProcessor",
]


if TYPE_CHECKING:
    # This significantly improves completion suggestion performance when
    # the transformers package is used with Microsoft's Pylance language server.
    IMAGE_PROCESSOR_MAPPING_NAMES: OrderedDict[str, dict[str, str | None]] = OrderedDict()
else:
    # Merge non-standard mapping names with auto-inferred `IMAGE_PROCESSOR_MAPPING_NAMES`
    MISSING_IMAGE_PROCESSOR_MAPPING_NAMES = OrderedDict(
        [
            ("aimv2", {"torchvision": "CLIPImageProcessor", "pil": "CLIPImageProcessorPil"}),
            ("aimv2_vision_model", {"torchvision": "CLIPImageProcessor", "pil": "CLIPImageProcessorPil"}),
            ("align", {"torchvision": "EfficientNetImageProcessor", "pil": "EfficientNetImageProcessorPil"}),
            ("altclip", {"torchvision": "CLIPImageProcessor", "pil": "CLIPImageProcessorPil"}),
            ("aya_vision", {"torchvision": "GotOcr2ImageProcessor", "pil": "GotOcr2ImageProcessorPil"}),
            ("blip-2", {"torchvision": "BlipImageProcessor", "pil": "BlipImageProcessorPil"}),
            ("clipseg", {"torchvision": "ViTImageProcessor", "pil": "ViTImageProcessorPil"}),
            ("colpali", {"torchvision": "SiglipImageProcessor", "pil": "SiglipImageProcessorPil"}),
            ("colqwen2", {"torchvision": "Qwen2VLImageProcessor", "pil": "Qwen2VLImageProcessorPil"}),
            ("convnextv2", {"torchvision": "ConvNextImageProcessor", "pil": "ConvNextImageProcessorPil"}),
            ("cvt", {"torchvision": "ConvNextImageProcessor", "pil": "ConvNextImageProcessorPil"}),
            ("data2vec-vision", {"torchvision": "BeitImageProcessor", "pil": "BeitImageProcessorPil"}),
            ("depth_anything", {"torchvision": "DPTImageProcessor", "pil": "DPTImageProcessorPil"}),
            ("dinat", {"torchvision": "ViTImageProcessor", "pil": "ViTImageProcessorPil"}),
            ("dinov2", {"torchvision": "BitImageProcessor", "pil": "BitImageProcessorPil"}),
            ("donut-swin", {"torchvision": "DonutImageProcessor", "pil": "DonutImageProcessorPil"}),
            ("edgetam", {"torchvision": "Sam2ImageProcessor"}),
            ("emu3", {"pil": "Emu3ImageProcessor"}),
            ("eomt_dinov3", {"torchvision": "EomtImageProcessor", "pil": "EomtImageProcessorPil"}),
            ("florence2", {"torchvision": "CLIPImageProcessor", "pil": "CLIPImageProcessorPil"}),
            ("focalnet", {"torchvision": "BitImageProcessor", "pil": "BitImageProcessorPil"}),
            ("gemma3n", {"torchvision": "SiglipImageProcessor", "pil": "SiglipImageProcessorPil"}),
            ("git", {"torchvision": "CLIPImageProcessor", "pil": "CLIPImageProcessorPil"}),
            ("groupvit", {"torchvision": "CLIPImageProcessor", "pil": "CLIPImageProcessorPil"}),
            ("hiera", {"torchvision": "BitImageProcessor", "pil": "BitImageProcessorPil"}),
            ("ijepa", {"torchvision": "ViTImageProcessor", "pil": "ViTImageProcessorPil"}),
            ("instructblip", {"torchvision": "BlipImageProcessor", "pil": "BlipImageProcessorPil"}),
            ("internvl", {"torchvision": "GotOcr2ImageProcessor", "pil": "GotOcr2ImageProcessorPil"}),
            ("kosmos-2", {"torchvision": "CLIPImageProcessor", "pil": "CLIPImageProcessorPil"}),
            ("kosmos-2.5", {"torchvision": "Kosmos2_5ImageProcessor", "pil": "Kosmos2_5ImageProcessorPil"}),
            ("layoutxlm", {"torchvision": "LayoutLMv2ImageProcessor", "pil": "LayoutLMv2ImageProcessorPil"}),
            ("lighton_ocr", {"torchvision": "PixtralImageProcessor", "pil": "PixtralImageProcessorPil"}),
            ("llava_next_video", {"torchvision": "LlavaNextImageProcessor", "pil": "LlavaNextImageProcessorPil"}),
            ("lw_detr", {"torchvision": "DeformableDetrImageProcessor", "pil": "DeformableDetrImageProcessorPil"}),
            ("metaclip_2", {"torchvision": "CLIPImageProcessor", "pil": "CLIPImageProcessorPil"}),
            ("mgp-str", {"torchvision": "ViTImageProcessor", "pil": "ViTImageProcessorPil"}),
            ("mistral3", {"torchvision": "PixtralImageProcessor", "pil": "PixtralImageProcessorPil"}),
            ("mlcd", {"torchvision": "CLIPImageProcessor", "pil": "CLIPImageProcessorPil"}),
            (
                "mm-grounding-dino",
                {
                    "torchvision": "GroundingDinoImageProcessor",
                    "pil": "GroundingDinoImageProcessorPil",
                },
            ),
            ("mobilevitv2", {"torchvision": "MobileViTImageProcessor", "pil": "MobileViTImageProcessorPil"}),
            ("omdet-turbo", {"torchvision": "DetrImageProcessor", "pil": "DetrImageProcessorPil"}),
            ("paligemma", {"torchvision": "SiglipImageProcessor", "pil": "SiglipImageProcessorPil"}),
            ("pixio", {"torchvision": "BitImageProcessor", "pil": "BitImageProcessorPil"}),
            ("pp_ocrv5_mobile_det", {"torchvision": "PPOCRV5ServerDetImageProcessor"}),
            ("pp_ocrv5_mobile_rec", {"torchvision": "PPOCRV5ServerRecImageProcessor"}),
            ("pvt_v2", {"torchvision": "PvtImageProcessor", "pil": "PvtImageProcessorPil"}),
            ("qianfan_ocr", {"torchvision": "GotOcr2ImageProcessor", "pil": "GotOcr2ImageProcessorPil"}),
            ("qwen2_5_omni", {"torchvision": "Qwen2VLImageProcessor", "pil": "Qwen2VLImageProcessorPil"}),
            ("qwen2_5_vl", {"torchvision": "Qwen2VLImageProcessor", "pil": "Qwen2VLImageProcessorPil"}),
            ("qwen3_5", {"torchvision": "Qwen2VLImageProcessor", "pil": "Qwen2VLImageProcessorPil"}),
            ("qwen3_5_moe", {"torchvision": "Qwen2VLImageProcessor", "pil": "Qwen2VLImageProcessorPil"}),
            ("qwen3_omni_moe", {"torchvision": "Qwen2VLImageProcessor", "pil": "Qwen2VLImageProcessorPil"}),
            ("qwen3_vl", {"torchvision": "Qwen2VLImageProcessor", "pil": "Qwen2VLImageProcessorPil"}),
            ("regnet", {"torchvision": "ConvNextImageProcessor", "pil": "ConvNextImageProcessorPil"}),
            ("resnet", {"torchvision": "ConvNextImageProcessor", "pil": "ConvNextImageProcessorPil"}),
            ("sam2_video", {"torchvision": "Sam2ImageProcessor"}),
            ("sam3_lite_text", {"torchvision": "Sam3ImageProcessor"}),
            ("sam3_tracker", {"torchvision": "Sam3ImageProcessor"}),
            ("sam3_tracker_video", {"torchvision": "Sam3ImageProcessor"}),
            ("sam3_video", {"torchvision": "Sam3ImageProcessor"}),
            ("sam_hq", {"torchvision": "SamImageProcessor", "pil": "SamImageProcessorPil"}),
            ("shieldgemma2", {"torchvision": "Gemma3ImageProcessor", "pil": "Gemma3ImageProcessorPil"}),
            ("slanet", {"torchvision": "SLANeXtImageProcessor"}),
            ("swiftformer", {"torchvision": "ViTImageProcessor", "pil": "ViTImageProcessorPil"}),
            ("swin", {"torchvision": "ViTImageProcessor", "pil": "ViTImageProcessorPil"}),
            ("swinv2", {"torchvision": "ViTImageProcessor", "pil": "ViTImageProcessorPil"}),
            ("t5gemma2", {"torchvision": "Gemma3ImageProcessor", "pil": "Gemma3ImageProcessorPil"}),
            ("t5gemma2_encoder", {"torchvision": "Gemma3ImageProcessor", "pil": "Gemma3ImageProcessorPil"}),
            ("table-transformer", {"torchvision": "DetrImageProcessor", "pil": "DetrImageProcessorPil"}),
            ("timesformer", {"pil": "VideoMAEImageProcessorPil", "torchvision": "VideoMAEImageProcessor"}),
            ("timm_wrapper", {"pil": "TimmWrapperImageProcessor"}),
            ("trocr", {"torchvision": "ViTImageProcessor", "pil": "ViTImageProcessorPil"}),
            ("udop", {"torchvision": "LayoutLMv3ImageProcessor", "pil": "LayoutLMv3ImageProcessorPil"}),
            ("upernet", {"torchvision": "SegformerImageProcessor", "pil": "SegformerImageProcessorPil"}),
            ("video_llava", {"pil": "VideoLlavaImageProcessor"}),
            ("vipllava", {"torchvision": "CLIPImageProcessor", "pil": "CLIPImageProcessorPil"}),
            ("vit_mae", {"torchvision": "ViTImageProcessor", "pil": "ViTImageProcessorPil"}),
            ("vit_msn", {"torchvision": "ViTImageProcessor", "pil": "ViTImageProcessorPil"}),
            ("vivit", {"torchvision": "VivitImageProcessor"}),
            ("xclip", {"torchvision": "CLIPImageProcessor", "pil": "CLIPImageProcessorPil"}),
        ]
    )

    IMAGE_PROCESSOR_MAPPING_NAMES.update(MISSING_IMAGE_PROCESSOR_MAPPING_NAMES)

IMAGE_PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, IMAGE_PROCESSOR_MAPPING_NAMES)


def get_image_processor_class_from_name(class_name: str):
    """Resolve an image processor class name to its class. Handles both base names (e.g. CLIPImageProcessor)
    and PIL backend names (e.g. CLIPImageProcessorPil). No recursion needed since names are direct."""
    if class_name == "BaseImageProcessorFast":
        # kept for backward compatibility - return TorchvisionBackend
        from ...image_processing_backends import TorchvisionBackend

        return TorchvisionBackend

    # First, check registered extra content (user-registered classes)
    for mapping in IMAGE_PROCESSOR_MAPPING._extra_content.values():
        for extractor_class in mapping.values():
            if isinstance(extractor_class, type) and getattr(extractor_class, "__name__", None) == class_name:
                return extractor_class

    # Check the mapping names - class names are either base (torchvision) or base+Pil (pil)
    for model_type, extractors_dict in IMAGE_PROCESSOR_MAPPING_NAMES.items():
        if class_name in extractors_dict.values():
            module_name = model_type_to_module_name(model_type)
            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    # Fallback: class may be in main init (e.g. when dep is missing, returns dummy)
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


def get_image_processor_config(
    pretrained_model_name_or_path: str | os.PathLike,
    cache_dir: str | os.PathLike | None = None,
    force_download: bool = False,
    proxies: dict[str, str] | None = None,
    token: bool | str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    **kwargs,
):
    """
    Loads the image processor configuration from a pretrained model image processor configuration.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~ProcessorMixin.save_pretrained`] method, e.g., `./my_model_directory/`.

        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        proxies (`dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `hf auth login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the image processor configuration from local files.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Dict`: The configuration of the image processor.

    Examples:

    ```python
    # Download configuration from huggingface.co and cache.
    image_processor_config = get_image_processor_config("google-bert/bert-base-uncased")
    # This model does not have a image processor config so the result will be an empty dict.
    image_processor_config = get_image_processor_config("FacebookAI/xlm-roberta-base")

    # Save a pretrained image processor locally and you can reload its config
    from transformers import AutoImageProcessor

    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_processor.save_pretrained("image-processor-test")
    image_processor_config = get_image_processor_config("image-processor-test")
    ```"""
    # Load with a priority given to the nested processor config, if available in repo
    resolved_processor_file = cached_file(
        pretrained_model_name_or_path,
        filename=PROCESSOR_NAME,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
    )
    resolved_image_processor_file = cached_file(
        pretrained_model_name_or_path,
        filename=IMAGE_PROCESSOR_NAME,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
        _raise_exceptions_for_gated_repo=False,
        _raise_exceptions_for_missing_entries=False,
    )

    # An empty list if none of the possible files is found in the repo
    if not resolved_image_processor_file and not resolved_processor_file:
        logger.info("Could not locate the image processor configuration file.")
        return {}

    # Load image_processor dict. Priority goes as (nested config if found -> image processor config)
    # We are downloading both configs because almost all models have a `processor_config.json` but
    # not all of these are nested. We need to check if it was saved recently as nested or if it is legacy style
    image_processor_dict = {}
    if resolved_processor_file is not None:
        processor_dict = safe_load_json_file(resolved_processor_file)
        if "image_processor" in processor_dict:
            image_processor_dict = processor_dict["image_processor"]

    if resolved_image_processor_file is not None and image_processor_dict is None:
        image_processor_dict = safe_load_json_file(resolved_image_processor_file)

    return image_processor_dict


def _resolve_backend(backend: str | None, use_fast: bool | None, base_class_name: str | None) -> str:
    """Resolve raw backend inputs to a concrete backend name ('torchvision' or 'pil').

    Handles, in order:
    - Deprecated ``use_fast`` flag: warns and converts to an explicit backend string when no
      explicit backend is given.
    - Explicit backend string: returned as-is.
    - None resolution: forces 'pil' for processors in DEFAULT_TO_PIL_BACKEND_IMAGE_PROCESSORS
      (Lanczos interpolation, unsupported by torchvision); otherwise picks 'torchvision' when
      available, falling back to 'pil'.
    """
    if use_fast is not None:
        logger.warning_once(
            "The `use_fast` parameter is deprecated and will be removed in a future version. "
            'Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.'
        )
        if backend is None:
            backend = "torchvision" if use_fast else "pil"

    if backend is None:
        if base_class_name in DEFAULT_TO_PIL_BACKEND_IMAGE_PROCESSORS:
            return "pil"
        return "torchvision" if is_torchvision_available() else "pil"

    return backend


def _load_class_with_fallback(mapping, backend):
    """
    Load an image processor class from a backend-to-class mapping, with fallback.

    Tries the requested backend first, then the opposite standard backend,
    then any remaining backends. Works with both string class names and resolved class objects.

    Unavailable backends are detected via DummyObject: classes whose required libraries are missing
    are represented as DummyObject subclasses (is_dummy=True). When the torchvision backend is
    missing but a PIL variant exists, _LazyModule transparently returns the PIL class with its own
    warning, so _load_class_with_fallback naturally receives a usable class without extra gating.

    Args:
        mapping: dict mapping backend names (str) to class names (str) or class objects (type).
        backend: the preferred backend name (e.g. "torchvision", "pil").

    Returns:
        The loaded class, or None if no class could be loaded.
    """
    backends_to_try = [backend] + [k for k in mapping if k != backend]

    for b in backends_to_try:
        value = mapping.get(b)
        if value is None:
            continue

        # Value can be a class object (from resolved mapping) or a string class name
        if isinstance(value, type):
            processor_class = value
        else:
            processor_class = get_image_processor_class_from_name(value)

        if processor_class is None or getattr(processor_class, "is_dummy", False):
            continue

        if b != backend:
            logger.warning_once(f"Requested {backend} backend is not available. Falling back to {b} backend.")
        return processor_class

    return None


def _find_mapping_for_image_processor(base_class_name: str) -> dict | None:
    """
    Find the backend->class mapping that contains base_class_name in its values.
    Returns the mapping dict (including any custom registered backends) or None.
    """

    def _value_matches(val, name: str) -> bool:
        if val is None:
            return False
        if isinstance(val, str):
            return val == name
        if isinstance(val, type):
            return getattr(val, "__name__", None) == name
        return False

    for mapping_dict in IMAGE_PROCESSOR_MAPPING_NAMES.values():
        if any(_value_matches(v, base_class_name) for v in mapping_dict.values()):
            return mapping_dict

    for content in IMAGE_PROCESSOR_MAPPING._extra_content.values():
        if any(_value_matches(v, base_class_name) for v in content.values()):
            return content

    return None


def _load_backend_class(base_class_name, backend, is_legacy_fast=False):
    """
    Load image processor class for a given backend. Uses the mapping from
    IMAGE_PROCESSOR_MAPPING when base_class_name is found in its values (so config
    overrides and custom backends are respected). Falls back to base+Pil convention
    for remote code / unknown processors.
    """
    mapping = _find_mapping_for_image_processor(base_class_name)
    if mapping is None:
        mapping = {
            "torchvision": base_class_name,
            "pil": base_class_name + "Pil",
        }
    processor_class = _load_class_with_fallback(mapping, backend)

    # For legacy Fast classes, try the original Fast class name as last resort
    if processor_class is None and is_legacy_fast:
        processor_class = get_image_processor_class_from_name(base_class_name + "Fast")

    return processor_class


def _resolve_auto_map_class_ref(auto_map, backend):
    """Extract the class reference string from an auto_map entry based on backend preference.

    Returns:
        A string that may be:
        - A simple class name (e.g. `"MyImageProcessor"`)
        - A Hub reference in the form `upstream_repo--path/to/file.py::ClassName`, where the part before
          `--` is the upstream repo ID (used for trust_remote_code resolution).
    """
    if isinstance(auto_map, dict):
        return auto_map.get(backend) or next(iter(auto_map.values()))
    if isinstance(auto_map, (list, tuple)):
        if backend == "torchvision" and len(auto_map) > 1 and auto_map[1] is not None:
            return auto_map[1]
        return auto_map[0]
    # Single string (legacy)
    return auto_map


@requires(backends=("vision",))
class AutoImageProcessor:
    r"""
    This is a generic image processor class that will be instantiated as one of the image processor classes of the
    library when created with the [`AutoImageProcessor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise OSError(
            "AutoImageProcessor is designed to be instantiated "
            "using the `AutoImageProcessor.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(IMAGE_PROCESSOR_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        r"""
        Instantiate one of the image processor classes of the library from a pretrained model vocabulary.

        The image processor class to instantiate is selected based on the `model_type` property of the config object
        (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it's
        missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Params:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained image_processor hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a image processor file saved using the
                  [`~image_processing_utils.ImageProcessingMixin.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path to a saved image processor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model image processor should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the image processor files and override the cached versions if
                they exist.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `hf auth login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            use_fast (`bool`, *optional*, defaults to `False`):
                **Deprecated**: Use `backend="torchvision"` instead. This parameter is kept for backward compatibility.
                Use a fast torchvision-based image processor if it is supported for a given model.
                If a fast image processor is not available for a given model, a normal numpy-based image processor
                is returned instead.
            backend (`str`, *optional*, defaults to `None`):
                The backend to use for image processing. Can be:
                - `None`: Automatically select the best available backend (torchvision if available, otherwise pil)
                - `"torchvision"`: Use Torchvision backend (GPU-accelerated, faster)
                - `"pil"`: Use PIL backend (portable, CPU-only)
                - Any custom backend name registered via `register()` method
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final image processor object. If `True`, then this
                functions returns a `Tuple(image_processor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not image processor attributes: i.e., the part of
                `kwargs` which has not been used to update `image_processor` and is otherwise ignored.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            image_processor_filename (`str`, *optional*, defaults to `"config.json"`):
                The name of the file in the model directory to use for the image processor config.
            kwargs (`dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are image processor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* image processor attributes is
                controlled by the `return_unused_kwargs` keyword parameter.

        <Tip>

        Passing `token=True` is required when you want to use a private model.

        </Tip>

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor

        >>> # Download image processor from huggingface.co and cache.
        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

        >>> # If image processor files are in a directory (e.g. image processor was saved using *save_pretrained('./test/saved_model/')*)
        >>> # image_processor = AutoImageProcessor.from_pretrained("./test/saved_model/")
        ```"""
        config = kwargs.pop("config", None)
        use_fast = kwargs.pop("use_fast", None)
        backend_kwarg = kwargs.pop("backend", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True

        # Resolve the image processor config filename
        if "image_processor_filename" in kwargs:
            image_processor_filename = kwargs.pop("image_processor_filename")
        elif is_timm_local_checkpoint(pretrained_model_name_or_path):
            image_processor_filename = CONFIG_NAME
        else:
            image_processor_filename = IMAGE_PROCESSOR_NAME

        # Load the image processor config

        try:
            config_dict, _ = ImageProcessingMixin.get_image_processor_dict(
                pretrained_model_name_or_path, image_processor_filename=image_processor_filename, **kwargs
            )
        except Exception as initial_exception:
            # Fallback for Hub TimmWrapper checkpoints (image processing in config.json, not preprocessor_config.json)
            try:
                config_dict, _ = ImageProcessingMixin.get_image_processor_dict(
                    pretrained_model_name_or_path, image_processor_filename=CONFIG_NAME, **kwargs
                )
            except Exception:
                raise initial_exception

            if not is_timm_config_dict(config_dict):
                raise initial_exception

        image_processor_type = config_dict.get("image_processor_type", None)
        image_processor_auto_map = None
        if "AutoImageProcessor" in config_dict.get("auto_map", {}):
            image_processor_auto_map = config_dict["auto_map"]["AutoImageProcessor"]

        # Backward compat: infer from feature extractor config
        if image_processor_type is None and image_processor_auto_map is None:
            feature_extractor_class = config_dict.pop("feature_extractor_type", None)
            if feature_extractor_class is not None:
                image_processor_type = feature_extractor_class.replace("FeatureExtractor", "ImageProcessor")
            if "AutoFeatureExtractor" in config_dict.get("auto_map", {}):
                feature_extractor_auto_map = config_dict["auto_map"]["AutoFeatureExtractor"]
                image_processor_auto_map = feature_extractor_auto_map.replace("FeatureExtractor", "ImageProcessor")

        # If not in image processor config, try the model config
        if image_processor_type is None and image_processor_auto_map is None:
            if not isinstance(config, PreTrainedConfig):
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )
            image_processor_type = getattr(config, "image_processor_type", None)
            if hasattr(config, "auto_map") and "AutoImageProcessor" in config.auto_map:
                image_processor_auto_map = config.auto_map["AutoImageProcessor"]

        # Derive base_class_name from image_processor_type
        is_legacy_fast = False
        base_class_name = None
        if image_processor_type is not None:
            is_legacy_fast = image_processor_type.endswith("Fast")
            base_class_name = image_processor_type[:-4] if is_legacy_fast else image_processor_type

        backend = _resolve_backend(backend_kwarg, use_fast, base_class_name)

        image_processor_class = None
        if base_class_name is not None:
            image_processor_class = _load_backend_class(base_class_name, backend, is_legacy_fast)

        # Handle remote code
        has_remote_code = image_processor_auto_map is not None
        has_local_code = image_processor_class is not None or type(config) in IMAGE_PROCESSOR_MAPPING
        explicit_local_code = has_local_code and not (
            image_processor_class or _load_class_with_fallback(IMAGE_PROCESSOR_MAPPING[type(config)], backend)
        ).__module__.startswith("transformers.")
        if has_remote_code:
            class_ref = _resolve_auto_map_class_ref(image_processor_auto_map, backend)
            upstream_repo = class_ref.split("--")[0] if "--" in class_ref else None
            trust_remote_code = resolve_trust_remote_code(
                trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code, upstream_repo
            )

        if has_remote_code and trust_remote_code and not explicit_local_code:
            image_processor_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
            _ = kwargs.pop("code_revision", None)
            image_processor_class.register_for_auto_class()
            return image_processor_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        elif image_processor_class is not None:
            return image_processor_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        # Last try: we use the IMAGE_PROCESSOR_MAPPING.
        elif type(config) in IMAGE_PROCESSOR_MAPPING:
            image_processor_mapping = IMAGE_PROCESSOR_MAPPING[type(config)]
            image_processor_class = _load_class_with_fallback(image_processor_mapping, backend)

            if image_processor_class is not None:
                return image_processor_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

            available = [k for k, v in image_processor_mapping.items() if v is not None]
            raise ValueError(f"Could not find image processor class. Available backends: {', '.join(available)}")
        raise ValueError(
            f"Unrecognized image processor in {pretrained_model_name_or_path}. Should have a "
            f"`image_processor_type` key in its {IMAGE_PROCESSOR_NAME} of {CONFIG_NAME}, or one of the following "
            f"`model_type` keys in its {CONFIG_NAME}: {', '.join(c for c in IMAGE_PROCESSOR_MAPPING_NAMES)}"
        )

    @staticmethod
    def register(
        config_class,
        slow_image_processor_class: type | None = None,
        fast_image_processor_class: type | None = None,
        image_processor_classes: dict[str, type] | None = None,
        exist_ok: bool = False,
    ):
        """
        Register a new image processor for this class.

        Args:
            config_class ([`PreTrainedConfig`]):
                The configuration corresponding to the model to register.
            slow_image_processor_class (`type`, *optional*):
                The PIL backend image processor class (deprecated, use `image_processor_classes={"pil": ...}`).
            fast_image_processor_class (`type`, *optional*):
                The Torchvision backend image processor class (deprecated, use `image_processor_classes={"torchvision": ...}`).
            image_processor_classes (`dict[str, type]`, *optional*):
                Dictionary mapping backend names to image processor classes. Allows registering custom backends.
                Example: `{"pil": MyPilProcessor, "torchvision": MyTorchvisionProcessor, "custom": MyCustomProcessor}`
            exist_ok (`bool`, *optional*, defaults to `False`):
                If `True`, allow overwriting existing registrations.
        """
        # Handle backward compatibility: convert old parameters to new format
        if image_processor_classes is None:
            image_processor_classes = {}
            if slow_image_processor_class is not None:
                image_processor_classes["pil"] = slow_image_processor_class
            if fast_image_processor_class is not None:
                image_processor_classes["torchvision"] = fast_image_processor_class

        if not image_processor_classes:
            raise ValueError(
                "You need to specify at least one image processor class. "
                "Use `image_processor_classes={'backend_name': ProcessorClass}` or the deprecated "
                "`slow_image_processor_class`/`fast_image_processor_class` parameters."
            )

        # Avoid resetting existing processors if we are passing partial updates
        if config_class in IMAGE_PROCESSOR_MAPPING._extra_content:
            existing_mapping = IMAGE_PROCESSOR_MAPPING[config_class]
            existing_mapping.update(image_processor_classes)
            image_processor_classes = existing_mapping

        # Validate that all classes are proper image processor classes
        from ...image_processing_utils import BaseImageProcessor

        for backend_key, processor_class in image_processor_classes.items():
            if processor_class is not None and not issubclass(processor_class, BaseImageProcessor):
                raise ValueError(
                    f"Image processor class for backend '{backend_key}' must inherit from `BaseImageProcessor`. "
                    f"Got: {processor_class}"
                )
        IMAGE_PROCESSOR_MAPPING.register(config_class, image_processor_classes, exist_ok=exist_ok)


__all__ = ["IMAGE_PROCESSOR_MAPPING", "AutoImageProcessor"]
