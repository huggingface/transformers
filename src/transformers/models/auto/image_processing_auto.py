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
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING

# Build the list of all image processors
from ...configuration_utils import PreTrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...image_processing_utils import BaseImageProcessor, ImageProcessingMixin
from ...utils import (
    CONFIG_NAME,
    IMAGE_PROCESSOR_NAME,
    PROCESSOR_NAME,
    cached_file,
    is_timm_config_dict,
    is_timm_local_checkpoint,
    is_vision_available,
    logging,
    safe_load_json_file,
)
from ...utils.import_utils import requires
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)


logger = logging.get_logger(__name__)

# These image processors use Lanczos interpolation, which is not supported by fast image processors.
# To avoid important differences in outputs, we default to using the python backend for these processors.
DEFAULT_TO_PYTHON_BACKEND_IMAGE_PROCESSORS = [
    "ChameleonImageProcessor",
    "FlavaImageProcessor",
    "Idefics3ImageProcessor",
    "SmolVLMImageProcessor",
]

if TYPE_CHECKING:
    # This significantly improves completion suggestion performance when
    # the transformers package is used with Microsoft's Pylance language server.
    IMAGE_PROCESSOR_MAPPING_NAMES: OrderedDict[str, str] = OrderedDict()
else:
    IMAGE_PROCESSOR_MAPPING_NAMES = OrderedDict(
        [
            ("aimv2", "CLIPImageProcessor"),
            ("aimv2_vision_model", "CLIPImageProcessor"),
            ("align", "EfficientNetImageProcessor"),
            ("altclip", "CLIPImageProcessor"),
            ("aria", "AriaImageProcessor"),
            ("aya_vision", "GotOcr2ImageProcessor"),
            ("beit", "BeitImageProcessor"),
            ("bit", "BitImageProcessor"),
            ("blip", "BlipImageProcessor"),
            ("blip-2", "BlipImageProcessor"),
            ("bridgetower", "BridgeTowerImageProcessor"),
            ("chameleon", "ChameleonImageProcessor"),
            ("chinese_clip", "ChineseCLIPImageProcessor"),
            ("clip", "CLIPImageProcessor"),
            ("clipseg", "ViTImageProcessor"),
            ("cohere2_vision", "Cohere2VisionImageProcessor"),
            ("colpali", "SiglipImageProcessor"),
            ("colqwen2", "Qwen2VLImageProcessor"),
            ("conditional_detr", "ConditionalDetrImageProcessor"),
            ("convnext", "ConvNextImageProcessor"),
            ("convnextv2", "ConvNextImageProcessor"),
            ("cvt", "ConvNextImageProcessor"),
            ("data2vec-vision", "BeitImageProcessor"),
            ("deepseek_vl", "DeepseekVLImageProcessor"),
            ("deepseek_vl_hybrid", "DeepseekVLHybridImageProcessor"),
            ("deformable_detr", "DeformableDetrImageProcessor"),
            ("deit", "DeiTImageProcessor"),
            ("depth_anything", "DPTImageProcessor"),
            ("depth_pro", "DepthProImageProcessor"),
            ("detr", "DetrImageProcessor"),
            ("dinat", "ViTImageProcessor"),
            ("dinov2", "BitImageProcessor"),
            ("dinov3_vit", "DINOv3ViTImageProcessor"),
            ("donut-swin", "DonutImageProcessor"),
            ("dpt", "DPTImageProcessor"),
            ("edgetam", "Sam2ImageProcessor"),
            ("efficientloftr", "EfficientLoFTRImageProcessor"),
            ("efficientnet", "EfficientNetImageProcessor"),
            ("emu3", "Emu3ImageProcessor"),
            ("eomt", "EomtImageProcessor"),
            ("ernie4_5_vl_moe", "Ernie4_5_VL_MoeImageProcessor"),
            ("flava", "FlavaImageProcessor"),
            ("florence2", "CLIPImageProcessor"),
            ("focalnet", "BitImageProcessor"),
            ("fuyu", "FuyuImageProcessor"),
            ("gemma3", "Gemma3ImageProcessor"),
            ("gemma3n", "SiglipImageProcessor"),
            ("git", "CLIPImageProcessor"),
            ("glm46v", "Glm46VImageProcessor"),
            ("glm4v", "Glm4vImageProcessor"),
            ("glm_image", "GlmImageImageProcessor"),
            ("glpn", "GLPNImageProcessor"),
            ("got_ocr2", "GotOcr2ImageProcessor"),
            ("grounding-dino", "GroundingDinoImageProcessor"),
            ("groupvit", "CLIPImageProcessor"),
            ("hiera", "BitImageProcessor"),
            ("idefics", "IdeficsImageProcessor"),
            ("idefics2", "Idefics2ImageProcessor"),
            ("idefics3", "Idefics3ImageProcessor"),
            ("ijepa", "ViTImageProcessor"),
            ("imagegpt", "ImageGPTImageProcessor"),
            ("instructblip", "BlipImageProcessor"),
            ("internvl", "GotOcr2ImageProcessor"),
            ("janus", "JanusImageProcessor"),
            ("kosmos-2", "CLIPImageProcessor"),
            ("kosmos-2.5", "Kosmos2_5ImageProcessor"),
            ("layoutlmv2", "LayoutLMv2ImageProcessor"),
            ("layoutlmv3", "LayoutLMv3ImageProcessor"),
            ("layoutxlm", "LayoutLMv2ImageProcessor"),
            ("levit", "LevitImageProcessor"),
            ("lfm2_vl", "Lfm2VlImageProcessor"),
            ("lightglue", "LightGlueImageProcessor"),
            ("lighton_ocr", "PixtralImageProcessor"),
            ("llama4", "Llama4ImageProcessor"),
            ("llava", "LlavaImageProcessor"),
            ("llava_next", "LlavaNextImageProcessor"),
            ("llava_next_video", "LlavaNextImageProcessor"),
            ("llava_onevision", "LlavaOnevisionImageProcessor"),
            ("lw_detr", "DeformableDetrImageProcessor"),
            ("mask2former", "Mask2FormerImageProcessor"),
            ("maskformer", "MaskFormerImageProcessor"),
            ("metaclip_2", "CLIPImageProcessor"),
            ("mgp-str", "ViTImageProcessor"),
            ("mistral3", "PixtralImageProcessor"),
            ("mlcd", "CLIPImageProcessor"),
            ("mllama", "MllamaImageProcessor"),
            ("mm-grounding-dino", "GroundingDinoImageProcessor"),
            ("mobilenet_v1", "MobileNetV1ImageProcessor"),
            ("mobilenet_v2", "MobileNetV2ImageProcessor"),
            ("mobilevit", "MobileViTImageProcessor"),
            ("mobilevitv2", "MobileViTImageProcessor"),
            ("nougat", "NougatImageProcessor"),
            ("omdet-turbo", "DetrImageProcessor"),
            ("oneformer", "OneFormerImageProcessor"),
            ("ovis2", "Ovis2ImageProcessor"),
            ("owlv2", "Owlv2ImageProcessor"),
            ("owlvit", "OwlViTImageProcessor"),
            ("paddleocr_vl", "PaddleOCRVLImageProcessor"),
            ("paligemma", "SiglipImageProcessor"),
            ("perceiver", "PerceiverImageProcessor"),
            ("perception_lm", "PerceptionLMImageProcessor"),
            ("phi4_multimodal", "Phi4MultimodalImageProcessor"),
            ("pix2struct", "Pix2StructImageProcessor"),
            ("pixio", "BitImageProcessor"),
            ("pixtral", "PixtralImageProcessor"),
            ("poolformer", "PoolFormerImageProcessor"),
            ("prompt_depth_anything", "PromptDepthAnythingImageProcessor"),
            ("pvt", "PvtImageProcessor"),
            ("pvt_v2", "PvtImageProcessor"),
            ("qwen2_5_omni", "Qwen2VLImageProcessor"),
            ("qwen2_5_vl", "Qwen2VLImageProcessor"),
            ("qwen2_vl", "Qwen2VLImageProcessor"),
            ("qwen3_omni_moe", "Qwen2VLImageProcessor"),
            ("qwen3_vl", "Qwen2VLImageProcessor"),
            ("regnet", "ConvNextImageProcessor"),
            ("resnet", "ConvNextImageProcessor"),
            ("rt_detr", "RTDetrImageProcessor"),
            ("sam", "SamImageProcessor"),
            ("sam2", "Sam2ImageProcessor"),
            ("sam2_video", "Sam2ImageProcessor"),
            ("sam3", "Sam3ImageProcessor"),
            ("sam3_tracker", "Sam3ImageProcessor"),
            ("sam3_tracker_video", "Sam3ImageProcessor"),
            ("sam3_video", "Sam3ImageProcessor"),
            ("sam_hq", "SamImageProcessor"),
            ("segformer", "SegformerImageProcessor"),
            ("seggpt", "SegGptImageProcessor"),
            ("shieldgemma2", "Gemma3ImageProcessor"),
            ("siglip", "SiglipImageProcessor"),
            ("siglip2", "Siglip2ImageProcessor"),
            ("smolvlm", "SmolVLMImageProcessor"),
            ("superglue", "SuperGlueImageProcessor"),
            ("superpoint", "SuperPointImageProcessor"),
            ("swiftformer", "ViTImageProcessor"),
            ("swin", "ViTImageProcessor"),
            ("swin2sr", "Swin2SRImageProcessor"),
            ("swinv2", "ViTImageProcessor"),
            ("t5gemma2", "Gemma3ImageProcessor"),
            ("table-transformer", "DetrImageProcessor"),
            ("textnet", "TextNetImageProcessor"),
            ("timesformer", "VideoMAEImageProcessor"),
            ("timm_wrapper", "TimmWrapperImageProcessor"),
            ("trocr", "ViTImageProcessor"),
            ("tvp", "TvpImageProcessor"),
            ("udop", "LayoutLMv3ImageProcessor"),
            ("upernet", "SegformerImageProcessor"),
            ("video_llama_3", "VideoLlama3ImageProcessor"),
            ("video_llava", "VideoLlavaImageProcessor"),
            ("videomae", "VideoMAEImageProcessor"),
            ("vilt", "ViltImageProcessor"),
            ("vipllava", "CLIPImageProcessor"),
            ("vit", "ViTImageProcessor"),
            ("vit_mae", "ViTImageProcessor"),
            ("vit_msn", "ViTImageProcessor"),
            ("vitmatte", "VitMatteImageProcessor"),
            ("vitpose", "VitPoseImageProcessor"),
            ("xclip", "CLIPImageProcessor"),
            ("yolos", "YolosImageProcessor"),
            ("zoedepth", "ZoeDepthImageProcessor"),
        ]
    )

# Override to None if vision is not available
if not is_vision_available():
    for model_type in list(IMAGE_PROCESSOR_MAPPING_NAMES.keys()):
        IMAGE_PROCESSOR_MAPPING_NAMES[model_type] = None

IMAGE_PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, IMAGE_PROCESSOR_MAPPING_NAMES)


def get_image_processor_class_from_name(class_name: str):
    if class_name == "BaseImageProcessorFast":
        # BaseImageProcessorFast has been unified with BaseImageProcessor
        return BaseImageProcessor

    base_class_name = class_name.removesuffix("Fast")
    is_fast_variant = class_name != base_class_name

    for module_name, processor_class_name in IMAGE_PROCESSOR_MAPPING_NAMES.items():
        if processor_class_name is None:
            continue
        if base_class_name == processor_class_name:
            module_name = model_type_to_module_name(module_name)
            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                if hasattr(module, processor_class_name):
                    return getattr(module, processor_class_name)
                # Fall back to Fast variant for backward compatibility
                elif is_fast_variant and hasattr(module, class_name):
                    return getattr(module, class_name)
            except AttributeError:
                continue

    for processor_class in IMAGE_PROCESSOR_MAPPING._extra_content.values():
        if isinstance(processor_class, tuple):
            # Legacy tuple format - check both entries
            for proc in processor_class:
                if proc is not None:
                    proc_name = getattr(proc, "__name__", None)
                    if proc_name == class_name or proc_name == base_class_name:
                        return proc
        elif processor_class is not None:
            proc_name = getattr(processor_class, "__name__", None)
            if proc_name == class_name or proc_name == base_class_name:
                return processor_class

    # We did not find the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # init and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)
    # Also try base class name in main module
    if is_fast_variant and hasattr(main_module, base_class_name):
        return getattr(main_module, base_class_name)

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
                - a path or url to a saved image processor JSON *file*, e.g.,
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
                **Deprecated.** Use `backend="torchvision"` instead.
                Use a fast torchvision-base image processor if it is supported for a given model.
                If a fast image processor is not available for a given model, a normal numpy-based image processor
                is returned instead.
            backend (`str`, *optional*, defaults to `"auto"`):
                Backend to use for image processing. Can be `"auto"`, `"python"`, or `"torchvision"`.
                - `"auto"`: Uses torchvision if available, otherwise python
                - `"torchvision"`: Uses GPU-accelerated TorchVision operations (faster, requires torchvision)
                - `"python"`: Uses NumPy/PIL operations (more portable, CPU-only)
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

        backend = kwargs.pop("backend", "auto")
        use_fast = kwargs.pop("use_fast", None)
        if use_fast is not None:
            warnings.warn(
                "The `use_fast` argument is deprecated and will be removed in v5 of Transformers. "
                "Use `backend='torchvision'` for fast processing or `backend='python'` for standard processing.",
                FutureWarning,
                stacklevel=2,
            )
            backend = "torchvision" if use_fast else "python"
        kwargs["backend"] = backend

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
            # Main path for all transformers models and local TimmWrapper checkpoints
            config_dict, _ = ImageProcessingMixin.get_image_processor_dict(
                pretrained_model_name_or_path, image_processor_filename=image_processor_filename, **kwargs
            )
        except Exception as initial_exception:
            # Fallback path for Hub TimmWrapper checkpoints. Timm models' image processing is saved in `config.json`
            # instead of `preprocessor_config.json`. Because this is an Auto class and we don't have any information
            # except the model name, the only way to check if a remote checkpoint is a timm model is to try to
            # load `config.json` and if it fails with some error, we raise the initial exception.
            try:
                config_dict, _ = ImageProcessingMixin.get_image_processor_dict(
                    pretrained_model_name_or_path, image_processor_filename=CONFIG_NAME, **kwargs
                )
            except Exception:
                raise initial_exception

            # In case we have a config_dict, but it's not a timm config dict, we raise the initial exception,
            # because only timm models have image processing in `config.json`.
            if not is_timm_config_dict(config_dict):
                raise initial_exception

        image_processor_type = config_dict.get("image_processor_type", None)
        image_processor_auto_map = None
        if "AutoImageProcessor" in config_dict.get("auto_map", {}):
            image_processor_auto_map = config_dict["auto_map"]["AutoImageProcessor"]

        # If we still don't have the image processor class, check if we're loading from a previous feature extractor config
        # and if so, infer the image processor class from there.
        if image_processor_type is None and image_processor_auto_map is None:
            feature_extractor_class = config_dict.pop("feature_extractor_type", None)
            if feature_extractor_class is not None:
                image_processor_type = feature_extractor_class.replace("FeatureExtractor", "ImageProcessor")
            if "AutoFeatureExtractor" in config_dict.get("auto_map", {}):
                feature_extractor_auto_map = config_dict["auto_map"]["AutoFeatureExtractor"]
                image_processor_auto_map = feature_extractor_auto_map.replace("FeatureExtractor", "ImageProcessor")

        # If we don't find the image processor class in the image processor config, let's try the model config.
        if image_processor_type is None and image_processor_auto_map is None:
            if not isinstance(config, PreTrainedConfig):
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )
            # It could be in `config.image_processor_type``
            image_processor_type = getattr(config, "image_processor_type", None)
            if hasattr(config, "auto_map") and "AutoImageProcessor" in config.auto_map:
                image_processor_auto_map = config.auto_map["AutoImageProcessor"]

        image_processor_class = None
        if image_processor_type is not None:
            # get_image_processor_class_from_name handles Fast variants for backward compatibility
            # (old configs/models may still reference Fast classes)
            image_processor_class = get_image_processor_class_from_name(image_processor_type)

        has_remote_code = image_processor_auto_map is not None
        has_local_code = image_processor_class is not None or type(config) in IMAGE_PROCESSOR_MAPPING
        if has_remote_code:
            # Handle both tuple (legacy) and single string formats
            if isinstance(image_processor_auto_map, (list, tuple)):
                class_ref = (
                    image_processor_auto_map[0]
                    if image_processor_auto_map[0] is not None
                    else image_processor_auto_map[1]
                )
            else:
                class_ref = image_processor_auto_map

            if "--" in class_ref:
                upstream_repo = class_ref.split("--")[0]
            else:
                upstream_repo = None
            trust_remote_code = resolve_trust_remote_code(
                trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code, upstream_repo
            )

        if has_remote_code and trust_remote_code:
            image_processor_class = get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs)
            _ = kwargs.pop("code_revision", None)
            image_processor_class.register_for_auto_class()
            return image_processor_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        elif image_processor_class is not None:
            return image_processor_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        # Last try: we use the IMAGE_PROCESSOR_MAPPING.
        elif type(config) in IMAGE_PROCESSOR_MAPPING:
            image_processor_class = IMAGE_PROCESSOR_MAPPING[type(config)]

            if image_processor_class is not None:
                return image_processor_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
            else:
                raise ValueError(
                    "This image processor cannot be instantiated. Please make sure you have `Pillow` installed."
                )
        raise ValueError(
            f"Unrecognized image processor in {pretrained_model_name_or_path}. Should have a "
            f"`image_processor_type` key in its {IMAGE_PROCESSOR_NAME} of {CONFIG_NAME}, or one of the following "
            f"`model_type` keys in its {CONFIG_NAME}: {', '.join(c for c in IMAGE_PROCESSOR_MAPPING_NAMES)}"
        )

    @staticmethod
    def register(
        config_class,
        image_processor_class=None,
        slow_image_processor_class=None,
        fast_image_processor_class=None,
        exist_ok=False,
    ):
        """
        Register a new image processor for this class.

        Args:
            config_class ([`PreTrainedConfig`]):
                The configuration corresponding to the model to register.
            image_processor_class ([`ImageProcessingMixin`]): The image processor to register.
                This is the preferred way to register. All image processors now support both backends.
            slow_image_processor_class ([`ImageProcessingMixin`], *optional*):
                **Deprecated.** Use `image_processor_class` instead. Kept for backward compatibility.
            fast_image_processor_class ([`ImageProcessingMixin`], *optional*):
                **Deprecated.** Use `image_processor_class` instead. Kept for backward compatibility.
        """

        # Handle backward compatibility with old API
        if image_processor_class is not None:
            if slow_image_processor_class is not None or fast_image_processor_class is not None:
                raise ValueError(
                    "Cannot specify both `image_processor_class` and `slow_image_processor_class`/`fast_image_processor_class`. "
                    "Use only `image_processor_class`."
                )
            processor_class = image_processor_class
        elif slow_image_processor_class is not None or fast_image_processor_class is not None:
            warnings.warn(
                "The `slow_image_processor_class` and `fast_image_processor_class` arguments are deprecated. "
                "All image processors now support both backends. Use `image_processor_class` instead.",
                FutureWarning,
                stacklevel=2,
            )
            # Prefer fast if both are provided, otherwise use whichever is available
            processor_class = (
                fast_image_processor_class if fast_image_processor_class is not None else slow_image_processor_class
            )
        else:
            raise ValueError("You need to specify `image_processor_class`.")

        if not issubclass(processor_class, BaseImageProcessor):
            raise ValueError("The `image_processor_class` should inherit from `BaseImageProcessor`.")

        # Check if already registered
        if config_class in IMAGE_PROCESSOR_MAPPING._extra_content:
            if not exist_ok:
                raise ValueError(f"`{config_class}` is already registered. Use `exist_ok=True` to override.")

        IMAGE_PROCESSOR_MAPPING.register(config_class, processor_class, exist_ok=exist_ok)


__all__ = ["IMAGE_PROCESSOR_MAPPING", "AutoImageProcessor"]
