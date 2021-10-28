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
    AutoModelForQuestionAnswering,
    TFAutoModelForQuestionAnswering,
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
            ov_model = load_ov_model_from_pytorch(model)
            ov_model.prepare_inputs_for_generation = model.prepare_inputs_for_generation
            ov_model._reorder_cache = model._reorder_cache
            return ov_model
        elif from_tf:
            model = TFAutoModelWithLMHead.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            return load_ov_model_from_tf(model)


class OVAutoModelForQuestionAnswering(object):
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
            model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            return load_ov_model_from_pytorch(model)
        elif from_tf:
            model = TFAutoModelForQuestionAnswering.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            return load_ov_model_from_tf(model)
        else:
            return OVAutoModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
