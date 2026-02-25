# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ... import initialization as init
from ...backbone_utils import BackboneMixin
from ...modeling_outputs import BackboneOutput, ImageClassifierOutput, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, can_return_tuple, is_timm_available, logging, requires_backends
from .configuration_timm_wrapper import TimmWrapperConfig


if is_timm_available():
    import timm

logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    Output class for models TimmWrapperModel, containing the last hidden states, an optional pooled output,
    and optional hidden states.
    """
)
class TimmWrapperModelOutput(ModelOutput):
    r"""
    last_hidden_state (`torch.FloatTensor`):
        The last hidden state of the model, output before applying the classification head.
    pooler_output (`torch.FloatTensor`, *optional*):
        The pooled output derived from the last hidden state, if applicable.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned if `output_hidden_states=True` is set or if `config.output_hidden_states=True`):
        A tuple containing the intermediate hidden states of the model at the output of each layer or specified layers.
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned if `output_attentions=True` is set or if `config.output_attentions=True`.):
        A tuple containing the intermediate attention weights of the model at the output of each layer.
        Note: Currently, Timm models do not support attentions output.
    """

    last_hidden_state: torch.FloatTensor
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


def _create_timm_model_with_error_handling(config: "TimmWrapperConfig", **model_kwargs):
    """
    Creates a timm model and provides a clear error message if the model is not found,
    suggesting a library update.
    """
    try:
        model = timm.create_model(
            config.architecture,
            pretrained=False,
            **model_kwargs,
        )
        return model
    except RuntimeError as e:
        if "Unknown model" in str(e):
            # A good general check for unknown models.
            raise ImportError(
                f"The model architecture '{config.architecture}' is not supported in your version of timm ({timm.__version__}). "
                "Please upgrade timm to a more recent version with `pip install -U timm`."
            ) from e
        raise e


@auto_docstring
class TimmWrapperPreTrainedModel(PreTrainedModel):
    base_model_prefix = "timm_model"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    config: TimmWrapperConfig
    # add WA here as `timm` does not support model parallelism
    _no_split_modules = ["TimmWrapperModel"]
    model_tags = ["timm"]

    # used in Trainer to avoid passing `loss_kwargs` to model forward
    accepts_loss_kwargs = False

    def post_init(self):
        self.supports_gradient_checkpointing = self._timm_model_supports_gradient_checkpointing()

        # Converts all `BatchNorm2d` and `SyncBatchNorm` or `BatchNormAct2d` and `SyncBatchNormAct2d` layers of
        # provided module into `FrozenBatchNorm2d` or `FrozenBatchNormAct2d` respectively
        if getattr(self.config, "freeze_batch_norm_2d", False):
            self.freeze_batch_norm_2d()

        super().post_init()

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Override original method to fix state_dict keys on load for cases when weights are loaded
        without using the `from_pretrained` method (e.g., in Trainer to resume from checkpoint).
        """
        state_dict = {f"timm_model.{k}" if "timm_model." not in k else k: v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, *args, **kwargs)

    @torch.no_grad()
    def _init_weights(self, module):
        """
        Initialize weights function to properly initialize Linear layer weights.
        Since model architectures may vary, we assume only the classifier requires
        initialization, while all other weights should be loaded from the checkpoint.
        """
        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                init.zeros_(module.bias)
        # Also, reinit all non-persistemt buffers if any!
        if hasattr(module, "init_non_persistent_buffers"):
            module.init_non_persistent_buffers()
        elif isinstance(module, nn.BatchNorm2d):
            # TimmWrapper always creates models with pretrained=False, so buffers are never pre-loaded
            # Always initialize buffers (handles both meta device and to_empty() cases)
            running_mean = getattr(module, "running_mean", None)
            if running_mean is not None:
                init.zeros_(module.running_mean)
                init.ones_(module.running_var)
                init.zeros_(module.num_batches_tracked)

    def _timm_model_supports_gradient_checkpointing(self):
        """
        Check if the timm model supports gradient checkpointing by checking if the `set_grad_checkpointing` method is available.
        Some timm models will have the method but will raise an AssertionError when called so in this case we return False.
        """
        if not hasattr(self.timm_model, "set_grad_checkpointing"):
            return False

        try:
            self.timm_model.set_grad_checkpointing(enable=True)
            self.timm_model.set_grad_checkpointing(enable=False)
            return True
        except Exception:
            return False

    def _set_gradient_checkpointing(self, enable: bool = True, *args, **kwargs):
        self.timm_model.set_grad_checkpointing(enable)

    def freeze_batch_norm_2d(self):
        timm.utils.model.freeze_batch_norm_2d(self.timm_model)

    def unfreeze_batch_norm_2d(self):
        timm.utils.model.unfreeze_batch_norm_2d(self.timm_model)

    def get_input_embeddings(self):
        # TIMM backbones operate directly on images and do not expose token embeddings.
        return None

    def set_input_embeddings(self, value):
        raise NotImplementedError("TimmWrapper models do not own token embeddings and cannot set them.")


class TimmWrapperBackboneModel(BackboneMixin, TimmWrapperPreTrainedModel):
    """
    Wrapper class for timm models to be used as backbones. This enables using the timm models interchangeably with the
    other models in the library keeping the same API.
    """

    def __init__(self, config, **kwargs):
        requires_backends(self, ["vision", "timm"])

        extra_init_kwargs = config.model_args or {}
        self.features_only = extra_init_kwargs.get("features_only", True)

        # We just take the final layer by default. This matches the default for the transformers models.
        out_indices = config.out_indices if getattr(config, "out_indices", None) is not None else (-1,)
        timm_backbone = _create_timm_model_with_error_handling(config, out_indices=out_indices, **extra_init_kwargs)

        # Needs to be called after creating timm model, because `super()` will try to infer
        # `stage_names` from model architecture
        super().__init__(config, timm_backbone=timm_backbone)
        self.timm_model = timm_backbone

        # These are used to control the output of the model when called. If output_hidden_states is True, then
        # return_layers is modified to include all layers.
        self._return_layers = {
            layer["module"]: str(layer["index"]) for layer in self.timm_model.feature_info.get_dicts()
        }
        self._all_layers = {layer["module"]: str(i) for i, layer in enumerate(self.timm_model.feature_info.info)}

        self.post_init()

    @property
    def _backbone(self):
        logger.warning(
            f"The `self._backbone` attribute is deprecated for {self.__class__.__name__}. Please use `self.timm_model` instead."
        )
        return self.timm_model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BackboneOutput | tuple[Tensor, ...]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        if output_attentions:
            raise ValueError("Cannot output attentions for timm backbones at the moment")

        if output_hidden_states:
            # We modify the return layers to include all the stages of the backbone
            self.timm_model.return_layers = self._all_layers
            hidden_states = self.timm_model(pixel_values)
            self.timm_model.return_layers = self._return_layers
            feature_maps = tuple(hidden_states[i] for i in self.out_indices)
        else:
            feature_maps = self.timm_model(pixel_values)
            hidden_states = None

        feature_maps = tuple(feature_maps)
        hidden_states = tuple(hidden_states) if hidden_states is not None else None

        return BackboneOutput(feature_maps=feature_maps, hidden_states=hidden_states, attentions=None)


class TimmWrapperModel(TimmWrapperPreTrainedModel):
    """
    Wrapper class for timm models to be used in transformers.
    """

    def __init__(self, config: TimmWrapperConfig):
        requires_backends(self, ["vision", "timm"])
        super().__init__(config)
        # using num_classes=0 to avoid creating classification head
        extra_init_kwargs = config.model_args or {}
        self.features_only = extra_init_kwargs.get("features_only", False)
        self.timm_model = _create_timm_model_with_error_handling(config, num_classes=0, **extra_init_kwargs)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = None,
        output_hidden_states: bool | list[int] | None = None,
        do_pooling: bool | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> TimmWrapperModelOutput | tuple[Tensor, ...]:
        r"""
        do_pooling (`bool`, *optional*):
            Whether to do pooling for the last_hidden_state in `TimmWrapperModel` or not. If `None` is passed, the
            `do_pooling` value from the config is used.

        Examples:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> from urllib.request import urlopen
        >>> from transformers import AutoModel, AutoImageProcessor

        >>> # Load image
        >>> image = Image.open(urlopen(
        ...     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
        ... ))

        >>> # Load model and image processor
        >>> checkpoint = "timm/resnet50.a1_in1k"
        >>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
        >>> model = AutoModel.from_pretrained(checkpoint).eval()

        >>> # Preprocess image
        >>> inputs = image_processor(image)

        >>> # Forward pass
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # Get pooled output
        >>> pooled_output = outputs.pooler_output

        >>> # Get last hidden state
        >>> last_hidden_state = outputs.last_hidden_state
        ```
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        do_pooling = do_pooling if do_pooling is not None else self.config.do_pooling

        if output_attentions:
            raise ValueError("Cannot set `output_attentions` for timm models.")

        if output_hidden_states and not hasattr(self.timm_model, "forward_intermediates"):
            raise ValueError(
                "The 'output_hidden_states' option cannot be set for this timm model. "
                "To enable this feature, the 'forward_intermediates' method must be implemented "
                "in the timm model (available in timm versions > 1.*). Please consider using a "
                "different architecture or updating the timm package to a compatible version."
            )

        pixel_values = pixel_values.to(self.device)

        if self.features_only:
            # TODO: ideally features only should be used with `BackboneModel`, deprecate here!
            last_hidden_state = self.timm_model.forward(pixel_values)
            hidden_states = last_hidden_state if output_hidden_states else None
            pooler_output = None
        else:
            if output_hidden_states:
                # to enable hidden states selection
                if isinstance(output_hidden_states, (list, tuple)):
                    kwargs["indices"] = output_hidden_states
                last_hidden_state, hidden_states = self.timm_model.forward_intermediates(pixel_values, **kwargs)
            else:
                last_hidden_state = self.timm_model.forward_features(pixel_values, **kwargs)
                hidden_states = None

            if do_pooling:
                # classification head is not created, applying pooling only
                pooler_output = self.timm_model.forward_head(last_hidden_state)
            else:
                pooler_output = None

        return TimmWrapperModelOutput(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=hidden_states,
        )


class TimmWrapperForImageClassification(TimmWrapperPreTrainedModel):
    """
    Wrapper class for timm models to be used in transformers for image classification.
    """

    def __init__(self, config: TimmWrapperConfig):
        requires_backends(self, ["vision", "timm"])
        super().__init__(config)

        if config.num_labels == 0:
            raise ValueError(
                "You are trying to load weights into `TimmWrapperForImageClassification` from a checkpoint with no classifier head. "
                "Please specify the number of classes, e.g. `model = TimmWrapperForImageClassification.from_pretrained(..., num_labels=10)`, "
                "or use `TimmWrapperModel` for feature extraction."
            )

        extra_init_kwargs = config.model_args or {}
        self.timm_model = _create_timm_model_with_error_handling(
            config, num_classes=config.num_labels, **extra_init_kwargs
        )
        self.num_labels = config.num_labels
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | list[int] | None = None,
        **kwargs,
    ) -> ImageClassifierOutput | tuple[Tensor, ...]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Examples:
        ```python
        >>> import torch
        >>> from PIL import Image
        >>> from urllib.request import urlopen
        >>> from transformers import AutoModelForImageClassification, AutoImageProcessor

        >>> # Load image
        >>> image = Image.open(urlopen(
        ...     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
        ... ))

        >>> # Load model and image processor
        >>> checkpoint = "timm/resnet50.a1_in1k"
        >>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
        >>> model = AutoModelForImageClassification.from_pretrained(checkpoint).eval()

        >>> # Preprocess image
        >>> inputs = image_processor(image)

        >>> # Forward pass
        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits

        >>> # Get top 5 predictions
        >>> top5_probabilities, top5_class_indices = torch.topk(logits.softmax(dim=1) * 100, k=5)
        ```
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        if output_attentions:
            raise ValueError("Cannot set `output_attentions` for timm models.")

        if output_hidden_states and not hasattr(self.timm_model, "forward_intermediates"):
            raise ValueError(
                "The 'output_hidden_states' option cannot be set for this timm model. "
                "To enable this feature, the 'forward_intermediates' method must be implemented "
                "in the timm model (available in timm versions > 1.*). Please consider using a "
                "different architecture or updating the timm package to a compatible version."
            )

        pixel_values = pixel_values.to(self.device, self.dtype)

        if output_hidden_states:
            # to enable hidden states selection
            if isinstance(output_hidden_states, (list, tuple)):
                kwargs["indices"] = output_hidden_states
            last_hidden_state, hidden_states = self.timm_model.forward_intermediates(pixel_values, **kwargs)
            logits = self.timm_model.forward_head(last_hidden_state)
        else:
            logits = self.timm_model(pixel_values, **kwargs)
            hidden_states = None

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )


__all__ = [
    "TimmWrapperBackboneModel",
    "TimmWrapperPreTrainedModel",
    "TimmWrapperModel",
    "TimmWrapperForImageClassification",
]
