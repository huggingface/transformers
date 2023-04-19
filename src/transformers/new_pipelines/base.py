from accelerate.state import PartialState
from accelerate.utils import send_to_device

from ..models.auto import AutoProcessor


class Tool:
    pre_processor_class = AutoProcessor
    model_class = None
    post_processor_class = AutoProcessor
    description = ""

    def __init__(
        self,
        model,
        pre_processor=None,
        post_processor=None,
        device=None,
        device_map=None,
        model_kwargs=None,
        **hub_kwargs,
    ):
        if pre_processor is None:
            pre_processor = model

        # Instantiate me maybe
        if isinstance(pre_processor, str):
            pre_processor = self.pre_processor_class.from_pretrained(pre_processor, **hub_kwargs)
        self.pre_processor = pre_processor

        if isinstance(model, str):
            if model_kwargs is None:
                model_kwargs = {}
            if device_map is not None:
                model_kwargs["device_map"] = device_map
            model = self.model_class.from_pretrained(model, **model_kwargs, **hub_kwargs)
        self.model = model

        if post_processor is None:
            post_processor = pre_processor
        elif isinstance(post_processor, str):
            post_processor = self.post_processor_class.from_pretrained(post_processor, **hub_kwargs)
        self.post_processor = post_processor

        if device is None:
            if device_map is not None:
                device = list(model.hf_device_map.values())[0]
            else:
                device = PartialState().default_device

        self.device = device
        if device_map is None:
            self.model.to(self.device)

    def encode(self, raw_inputs):
        return self.pre_processor(raw_inputs)

    def forward(self, inputs):
        return self.model(**inputs)

    def decode(self, outputs):
        return self.post_processor(outputs)

    def __call__(self, *args, **kwargs):
        encoded_inputs = self.encode(*args, **kwargs)
        encoded_inputs = send_to_device(encoded_inputs, self.device)
        outputs = self.forward(encoded_inputs)
        outputs = send_to_device(outputs, "cpu")
        return self.decode(outputs)
