import importlib
import json
from typing import List

from huggingface_hub import InferenceApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

from ..dynamic_module_utils import get_class_from_dynamic_module
from ..models.auto import AutoProcessor
from ..utils import CONFIG_NAME, cached_file, is_accelerate_available, is_torch_available


if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate.utils import send_to_device


TOOL_CONFIG_FILE = "tool_config.json"


class Tool:
    """
    Example of a super 'Tool' class that could live in huggingface_hub
    """

    description = "This is a tool that ..."
    is_initialized = False

    inputs: List[str]
    outputs: List[str]
    name: str

    def __init__(self, *args, **kwargs):
        pass

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
        token=None,
        **hub_kwargs,
    ):
        if not is_torch_available():
            raise ImportError("Please install torch in order to use this tool.")

        if not is_accelerate_available():
            raise ImportError("Please install accelerate in order to use this tool.")

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
        self.hub_kwargs["use_auth_token"] = token

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
    if not is_torch_available():
        raise ImportError("Please install torch in order to use this tool.")

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


TASK_MAPPING = {
    "generative-qa": "GenerativeQuestionAnsweringTool",
    "image-transformation": "ImageTransformationTool",
    "image-captioning": "ImageCaptioningTool",
    "image-generation": "TextToImageTool",
    "image-segmentation": "ImageSegmentationTool",
    "language-identification": "LanguageIdenticationTool",
    "speech-to-text": "SpeechToTextTool",
    "text-classification": "TextClassificationTool",
    "text-to-speech": "TextToSpeechTool",
    "translation": "TranslationTool",
    "text-download": "TextDownloadTool",
    "summarizer": "TextSummarizationTool",
    "image-question-answering": "ImageQuestionAnsweringTool",
    "document-question-answering": "DocumentQuestionAnsweringTool",
    "text-to-video": "TextToVideoTool",
}


def supports_remote(task_name):
    if task_name not in TASK_MAPPING:
        return False
    main_module = importlib.import_module("transformers")
    tools_module = main_module.tools
    tool_class = TASK_MAPPING[task_name]
    return hasattr(tools_module, f"Remote{tool_class}")


def get_repo_type(repo_id, repo_type=None, **hub_kwargs):
    if repo_type is not None:
        return repo_type
    try:
        hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type="space", **hub_kwargs)
        return "space"
    except RepositoryNotFoundError:
        try:
            hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type="model", **hub_kwargs)
            return "model"
        except RepositoryNotFoundError:
            raise EnvironmentError(f"`{repo_id}` does not seem to be a valid repo identifier on the Hub.")
        except Exception:
            return "model"
    except Exception:
        return "space"


def tool(task_or_repo_id, repo_id=None, model_repo_id=None, remote=False, token=None, **tool_kwargs):
    # Make sure to keep this list updated with the doc of tool_kwargs (when it exists lol)
    hub_kwargs_names = [
        "cache_dir",
        "force_download",
        "resume_download",
        "proxies",
        "revision",
        "repo_type",
        "subfolder",
        "local_files_only",
    ]
    hub_kwargs = {k: v for k, v in tool_kwargs.items() if k in hub_kwargs_names}

    if task_or_repo_id in TASK_MAPPING:
        tool_class_name = TASK_MAPPING[task_or_repo_id]
        if remote:
            if not supports_remote(task_or_repo_id):
                raise NotImplementedError(
                    f"{task_or_repo_id} does not support the inference API or inference endpoints yet."
                )
            tool_class_name = f"Remote{tool_class_name}"

        main_module = importlib.import_module("transformers")
        tools_module = main_module.tools
        tool_class = getattr(tools_module, tool_class_name)
    else:
        if repo_id is None:
            repo_id = task_or_repo_id
            task = None
        else:
            task = task_or_repo_id

        # Try to get the tool config first.
        hub_kwargs["repo_type"] = get_repo_type(repo_id, **hub_kwargs)
        resolved_config_file = cached_file(
            repo_id,
            TOOL_CONFIG_FILE,
            use_auth_token=token,
            **hub_kwargs,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        is_tool_config = resolved_config_file is not None
        if resolved_config_file is None:
            resolved_config_file = cached_file(
                repo_id,
                CONFIG_NAME,
                use_auth_token=token,
                **hub_kwargs,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
        if resolved_config_file is None:
            raise EnvironmentError(
                f"{repo_id} does not appear to provide a valid configuration in `tool_config.json` or `config.json`."
            )

        with open(resolved_config_file, encoding="utf-8") as reader:
            config = json.load(reader)

        if not is_tool_config:
            if "custom_tools" not in config:
                raise EnvironmentError(
                    f"{repo_id} does not provide a mapping to custom tools in its configuration `config.json`."
                )
            custom_tools = config["custom_tools"]
        else:
            custom_tools = config
        if task is None:
            if len(custom_tools) == 1:
                task = list(custom_tools.keys())[0]
            else:
                tasks_available = "\n".join([f"- {t}" for t in custom_tools.keys()])
                raise ValueError(f"Please select a task among the one available in {repo_id}:\n{tasks_available}")

        tool_class = get_class_from_dynamic_module(
            custom_tools[task]["tool_class"], repo_id, use_auth_token=token, **hub_kwargs
        )
        if model_repo_id is not None:
            repo_id = model_repo_id
        elif hub_kwargs["repo_type"] == "space":
            repo_id = None

    return tool_class(repo_id, token=token, **tool_kwargs)


def add_description(description):
    """
    A decorator that adds a description to a function.
    """

    def inner(func):
        func.description = description
        return func

    return inner
