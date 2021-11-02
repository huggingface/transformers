from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelWithLMHead,
    TFAutoModel,
    TFAutoModelForMaskedLM,
    TFAutoModelForQuestionAnswering,
    TFAutoModelForSequenceClassification,
    TFAutoModelWithLMHead,
)

from ...modeling_ov_utils import OVPreTrainedModel


class _BaseOVAutoModelClass(OVPreTrainedModel):
    def __init__(self):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` method."
        )


class OVAutoModel(_BaseOVAutoModelClass):
    _pt_auto_model = AutoModel
    _tf_auto_model = TFAutoModel


class OVAutoModelForMaskedLM(_BaseOVAutoModelClass):
    _pt_auto_model = AutoModelForMaskedLM
    _tf_auto_model = TFAutoModelForMaskedLM


class OVAutoModelWithLMHead(_BaseOVAutoModelClass):
    _pt_auto_model = AutoModelWithLMHead
    _tf_auto_model = TFAutoModelWithLMHead


class OVAutoModelForQuestionAnswering(_BaseOVAutoModelClass):
    _pt_auto_model = AutoModelForQuestionAnswering
    _tf_auto_model = TFAutoModelForQuestionAnswering


class OVAutoModelForSequenceClassification(_BaseOVAutoModelClass):
    _pt_auto_model = AutoModelForSequenceClassification
    _tf_auto_model = TFAutoModelForSequenceClassification
