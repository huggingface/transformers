from typing import List

import torch
from accelerate.utils import send_to_device
from huggingface_hub import InferenceApi

import transformers

from ..models.auto import AutoProcessor


class Tool:
    """
    Example of a super 'Tool' class that could live in huggingface_hub
    """

    description = "This is a tool that ..."
    is_initialized = False

    inputs: List[str]
    outputs: List[str]
    name: str

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


class RemoteTool(Tool):
    default_checkpoint = None
    description = "This is a tool that ..."

    def __init__(self, repo_id=None, token=None):
        if repo_id is None:
            repo_id = self.default_checkpoint
        self.repo_id = repo_id
        self.client = InferenceApi(repo_id, token=token)
        self.post_init()

    def prepare_inputs(self, *args, **kwargs):
        if len(args) > 1:
            raise ValueError("A `RemoteTool` can only accept one positional input.")
        elif len(args) == 1:
            return {"data": args[0]}

        return {"inputs": kwargs}

    def extract_outputs(self, outputs):
        return outputs

    def __call__(self, *args, **kwargs):
        if not self.is_initialized:
            self.setup()

        inputs = self.prepare_inputs(*args, **kwargs)
        if isinstance(inputs, dict):
            outputs = self.client(**inputs)
        else:
            outputs = self.client(inputs)
        if isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], list):
            outputs = outputs[0]
        return self.extract_outputs(outputs)


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

        self.is_initialized = False
        self.post_init()

    def setup(self):
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
                self.device = get_default_device()

        if self.device_map is None:
            self.model.to(self.device)

    def post_init(self):
        pass

    def encode(self, raw_inputs):
        return self.pre_processor(raw_inputs)

    def forward(self, inputs):
        return self.model(**inputs)

    def decode(self, outputs):
        return self.post_processor(outputs)

    def __call__(self, *args, **kwargs):
        if not self.is_initialized:
            self.setup()

        encoded_inputs = self.encode(*args, **kwargs)
        encoded_inputs = send_to_device(encoded_inputs, self.device)
        outputs = self.forward(encoded_inputs)
        outputs = send_to_device(outputs, "cpu")
        return self.decode(outputs)


def launch_gradio_demo(tool_class: Tool):
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio should be installed in order to launch a gradio demo.")

    tool = tool_class()

    def fn(*args, **kwargs):
        return tool(*args, **kwargs)

    gr.Interface(
        fn=fn,
        inputs=tool_class.inputs,
        outputs=tool_class.outputs,
        title=tool_class.__name__,
        article=tool.description,
    ).launch()


# TODO: Migrate to Accelerate for this once `PartialState.default_device` makes its way into a release.
def get_default_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


TASK_MAPPING = {
    "generative_qa": "GenerativeQuestionAnsweringTool",
    "image_alteration": "ControlNetTool",
    "image_captioning": "ImageCaptioningTool",
    "image_generation": "StableDiffusionTool",
    "image_segmentation": "ImageSegmentationTool",
    "language_identification": "LanguageIdenticationTool",
    "speech_to_text": "SpeechToTextTool",
    "text_classification": "TextClassificationTool",
    "text_to_speech": "TextToSpeechTool",
    "translation": "TranslationTool",
}


def tool(task_or_repo_id, repo_id=None, remote=False, token=None, **tool_kwargs):
    task_or_repo_id = task_or_repo_id.replace("-", "_")
    if task_or_repo_id in TASK_MAPPING:
        tool_class_name = TASK_MAPPING[task_or_repo_id]
        if remote:
            tool_class_name = f"Remote{tool_class_name}"
        try:
            tool_class = getattr(transformers.tools, tool_class_name)
        except ImportError:
            if remote:
                raise NotImplementedError(
                    f"{task_or_repo_id} does not support the inference API or inference endpoints yet."
                )
            raise
    else:
        repo_id = task_or_repo_id
        # TODO: code on the hub for the tool
        raise NotImplementedError("Support for code on the Hub is coming soon!")

    return tool_class(repo_id, token=token, **tool_kwargs)
