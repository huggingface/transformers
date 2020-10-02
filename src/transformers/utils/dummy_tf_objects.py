from ..file_utils import requires_tf


class TFBertModel:
    def __init__(self, *args, **kwargs):
        requires_tf(self)

    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_tf(self)
