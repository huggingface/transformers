from ...modeling_ov_utils import OVPreTrainedModel


class OVAutoModel(object):
    def __init__(self):
        print("init")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        return OVPreTrainedModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
