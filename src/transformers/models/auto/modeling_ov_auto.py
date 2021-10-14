from ...modeling_ov_utils import (
    OVPreTrainedModel,
    load_ov_model_from_tf,
    load_ov_model_from_pytorch,
)

from transformers import (
    AutoModelForMaskedLM,
    TFAutoModelForMaskedLM,
    AutoModelWithLMHead,
    TFAutoModelWithLMHead,
)


class OVAutoModel(object):
    def __init__(self):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        return OVPreTrainedModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class OVAutoModelForMaskedLM(object):
    def __init__(self):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        from_pt = kwargs.pop("from_pt", False)
        from_tf = kwargs.pop("from_tf", False)
        if from_pt:
            model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            return load_ov_model_from_pytorch(model)
        elif from_tf:
            model = TFAutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            return load_ov_model_from_tf(model)


class OVAutoModelWithLMHead(object):
    def __init__(self):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        from_pt = kwargs.pop("from_pt", False)
        from_tf = kwargs.pop("from_tf", False)
        if from_pt:
            model = AutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            return load_ov_model_from_pytorch(model)
        elif from_tf:
            model = TFAutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            return load_ov_model_from_tf(model)
