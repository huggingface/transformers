import warnings

from ..modeling_backbone_utils import BackboneConfigMixin, BackboneMixin


class BackboneConfigMixin(BackboneConfigMixin):
    warnings.warn(
        "Importing `BackboneConfigMixin` from `utils/backbone_utils.py` is deprecated and will be removed in "
        "Transformers v5.10. Import as `from transformers.modeling_backbone_utils import BackboneConfigMixin` instead.",
        FutureWarning,
    )


class BackboneMixin(BackboneMixin):
    warnings.warn(
        "Importing `BackboneMixin` from `utils/backbone_utils.py` is deprecated and will be removed in "
        "Transformers v5.10. Import as `from transformers.modeling_backbone_utils import BackboneMixin` instead.",
        FutureWarning,
    )
