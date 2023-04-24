from accelerate.state import PartialState
from accelerate.utils import send_to_device
from huggingface_hub import InferenceApi

from ..models.auto import AutoProcessor


class Tool:
    """
    Example of a super 'Tool' class that could live in huggingface_hub
    """

    description = "This is a tool that ..."

    def __call__(self, *args, **kwargs):  # Might become run?
        return NotImplemented("Write this method in your subclass of `Tool`.")

    def post_init(self):
        # Do here everything you need to execute after the init (to avoir overriding the init which is complex), such
        # as formatting the description with the attributes of your tools.
        pass

    def setup(self):
        # Do here any operation that is expensive and needs to be executed before you start using your tool. Such as
        # loading a big model.
        self.is_initialized = True


class PipelineTool(Tool):
    pre_processor_class = AutoProcessor
    model_class = None
    post_processor_class = AutoProcessor
    default_checkpoint = None
    description = "This is a tool that ..."

    def __init__(
        self,
        model=None,
        pre_processor=None,
        post_processor=None,
        device=None,
        device_map=None,
        model_kwargs=None,
        **hub_kwargs,
    ):
        if model is None:
            if self.default_checkpoint is None:
                raise ValueError("This tool does not implement a default checkpoint, you need to pass one.")
            model = self.default_checkpoint
        if pre_processor is None:
            pre_processor = model

        self.model = model
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        self.device = device
        self.device_map = device_map
        self.model_kwargs = {} if model_kwargs is None else model_kwargs
        if device_map is not None:
            self.model_kwargs["device_map"] = device_map
        self.hub_kwargs = hub_kwargs

        self.inference_api_mode = self.device == "hub"
        if self.inference_api_mode and not isinstance(self.model, str):
            raise ValueError("To use this tool in inference API mode, you need to provide a valid repo ID.")

        self.is_initialized = False
        self.post_init()

    def setup(self):
        if self.inference_api_mode:
            self.model = InferenceApi(self.model)
            return

        # Instantiate me maybe
        if isinstance(self.pre_processor, str):
            self.pre_processor = self.pre_processor_class.from_pretrained(self.pre_processor, **self.hub_kwargs)

        if isinstance(self.model, str):
            self.model = self.model_class.from_pretrained(self.model, **self.model_kwargs, **self.hub_kwargs)

        if self.post_processor is None:
            self.post_processor = self.pre_processor
        elif isinstance(self.post_processor, str):
            self.post_processor = self.post_processor_class.from_pretrained(self.post_processor, **self.hub_kwargs)

        if self.device is None:
            if self.device_map is not None:
                self.device = list(self.model.hf_device_map.values())[0]
            else:
                self.device = PartialState().default_device

        if self.device_map is None:
            self.model.to(self.device)

    def post_init(self):
        pass

    def encode(self, raw_inputs):
        return self.pre_processor(raw_inputs)

    def encode_for_hub(self, *args, **kwargs):
        # Deal with images here.
        return kwargs

    def forward(self, inputs):
        return self.model(**inputs)

    def decode(self, outputs):
        return self.post_processor(outputs)

    def decode_for_hub(outputs):
        return outputs

    def __call__(self, *args, **kwargs):
        if not self.is_initialized:
            self.setup()

        if self.inference_api_mode:
            inputs = self.encode_for_hub(*args, **kwargs)
            outputs = self.model(inputs)
            return self.decode_for_hub(outputs)

        encoded_inputs = self.encode(*args, **kwargs)
        encoded_inputs = send_to_device(encoded_inputs, self.device)
        outputs = self.forward(encoded_inputs)
        outputs = send_to_device(outputs, "cpu")
        return self.decode(outputs)
