import importlib
import io
import json
import os
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import CommitOperationAdd, InferenceApi, create_commit, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, build_hf_headers, get_session

from ..dynamic_module_utils import custom_object_save, get_class_from_dynamic_module, get_imports
from ..models.auto import AutoProcessor
from ..utils import (
    CONFIG_NAME,
    cached_file,
    is_accelerate_available,
    is_torch_available,
    is_vision_available,
    logging,
    working_or_temp_dir,
)


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate.utils import send_to_device


TOOL_CONFIG_FILE = "tool_config.json"


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


APP_FILE_TEMPLATE = """from transformers.tools.base import launch_gradio_demo from {module_name} import {class_name}

launch_gradio_demo({class_name})
"""


class Tool:
    """
    Example of a super 'Tool' class that could live in huggingface_hub
    """

    description: str = "This is a tool that ..."
    name: str = ""

    inputs: List[str]
    outputs: List[str]

    def __init__(self, *args, **kwargs):
        self.is_initialized = False
        pass

    def __call__(self, *args, **kwargs):  # Might become run?
        return NotImplemented("Write this method in your subclass of `Tool`.")

    def setup(self):
        # Do here any operation that is expensive and needs to be executed before you start using your tool. Such as
        # loading a big model.
        self.is_initialized = True

    def save(self, output_dir, task_name=None):
        os.makedirs(output_dir, exist_ok=True)
        # Save module file
        module_files = custom_object_save(self, output_dir)

        module_name = self.__class__.__module__
        last_module = module_name.split(".")[-1]
        full_name = f"{last_module}.{self.__class__.__name__}"

        # Save config file
        config_file = os.path.join(output_dir, "tool_config.json")
        if os.path.isfile(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                tool_config = json.load(f)
        else:
            tool_config = {}

        if task_name is None:
            class_name = self.__class__.__name__.replace("Tool", "")
            chars = [f"_{c.lower()}" if c.isupper() else c for c in class_name]
            task_name = "".join(chars)[1:]

        tool_config[task_name] = {"tool_class": full_name, "description": self.description, "name": self.name}
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(tool_config, indent=2, sort_keys=True) + "\n")

        # Save app file
        app_file = os.path.join(output_dir, "app.py")
        with open(app_file, "w", encoding="utf-8") as f:
            f.write(APP_FILE_TEMPLATE.format(module_name=last_module, class_name=self.__class__.__name__))

        # Save requirements file
        requirements_file = os.path.join(output_dir, "requirements.txt")
        imports = []
        for module in module_files:
            imports.extend(get_imports(module))
        imports = list(set(imports))
        with open(requirements_file, "w", encoding="utf-8") as f:
            f.write("\n".join(imports) + "\n")

    @classmethod
    def from_hub(cls, task_or_repo_id, repo_id=None, model_repo_id=None, token=None, remote=False, **kwargs):
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
        hub_kwargs = {k: v for k, v in kwargs.items() if k in hub_kwargs_names}
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

        tool_class = custom_tools[task]["tool_class"]
        if isinstance(tool_class, (list, tuple)):
            tool_class = tool_class[1] if remote else tool_class[0]
        tool_class = get_class_from_dynamic_module(tool_class, repo_id, use_auth_token=token, **hub_kwargs)
        if model_repo_id is not None:
            repo_id = model_repo_id
        elif hub_kwargs["repo_type"] == "space":
            repo_id = None

        return tool_class(repo_id, token=token, **kwargs)

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: str = "Upload tool",
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
    ) -> str:
        """
        Upload the {object_files} to the 🤗 Model Hub while synchronizing a local clone of the repo in
        `repo_path_or_name`.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your tool to. It should contain your organization name when
                pushing to a given organization.
            use_temp_dir (`bool`, *optional*):
                Whether or not to use a temporary directory to store the files saved before they are pushed to the Hub.
                Will default to `True` if there is no directory named like `repo_id`, `False` otherwise.
            commit_message (`str`, *optional*, defaults to `"Upload too"`):
                Message to commit while pushing.
            private (`bool`, *optional*):
                Whether or not the repository created should be private.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unsel, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
        """
        if os.path.isdir(repo_id):
            working_dir = repo_id
            repo_id = repo_id.split(os.path.sep)[-1]
        else:
            working_dir = repo_id.split("/")[-1]

        repo_url = create_repo(
            repo_id=repo_id, token=token, private=private, exist_ok=True, repo_type="space", space_sdk="gradio"
        )
        repo_id = repo_url.repo_id

        if use_temp_dir is None:
            use_temp_dir = not os.path.isdir(working_dir)

        with working_or_temp_dir(working_dir=working_dir, use_temp_dir=use_temp_dir) as work_dir:
            files_timestamps = self._get_files_timestamps(work_dir)

            # Save all files.
            self.save(work_dir)

            modified_files = [
                f
                for f in os.listdir(work_dir)
                if f not in files_timestamps or os.path.getmtime(os.path.join(work_dir, f)) > files_timestamps[f]
            ]
            operations = []
            for file in modified_files:
                operations.append(CommitOperationAdd(path_or_fileobj=os.path.join(work_dir, file), path_in_repo=file))
            logger.info(f"Uploading the following files to {repo_id}: {','.join(modified_files)}")
            return create_commit(
                repo_id=repo_id,
                operations=operations,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
                repo_type="space",
            )


class OldRemoteTool(Tool):
    default_checkpoint = None

    def __init__(self, repo_id=None, token=None):
        if repo_id is None:
            repo_id = self.default_checkpoint
        self.repo_id = repo_id
        self.client = InferenceApi(repo_id, token=token)

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


class RemoteTool(Tool):
    default_url = None

    def __init__(self, endpoint_url=None, token=None):
        if endpoint_url is None:
            endpoint_url = self.default_url
        self.endpoint_url = endpoint_url
        self.client = EndpointClient(endpoint_url, token=token)

    def prepare_inputs(self, *args, **kwargs):
        if len(args) > 1:
            raise ValueError("A `RemoteTool` can only accept one positional input.")
        elif len(args) == 1:
            return {"data": args[0]}

        return {"inputs": kwargs}

    def extract_outputs(self, outputs):
        return outputs

    def __call__(self, *args, **kwargs):
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
    "image-captioning": "ImageCaptioningTool",
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
}


def load_tool(task_or_repo_id, repo_id=None, remote=False, token=None, **kwargs):
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

        return tool_class(repo_id, token=token, **kwargs)
    else:
        return Tool.from_hub(task_or_repo_id, repo_id=repo_id, token=token, remote=remote, **kwargs)


def add_description(description):
    """
    A decorator that adds a description to a function.
    """

    def inner(func):
        func.description = description
        return func

    return inner


## Will move to the Hub
class EndpointClient:
    def __init__(self, endpoint_url: str, token: Optional[str] = None):
        self.headers = build_hf_headers(token=token)
        self.endpoint_url = endpoint_url

    def __call__(
        self,
        inputs: Optional[Union[str, Dict, List[str], List[List[str]]]] = None,
        params: Optional[Dict] = None,
        data: Optional[bytes] = None,
        raw_response: bool = False,
    ) -> Any:
        # Build payload
        payload = {}
        if inputs:
            payload["inputs"] = inputs
        if params:
            payload["parameters"] = params

        # Make API call
        response = get_session().post(self.endpoint_url, headers=self.headers, json=payload, data=data)

        # Let the user handle the response
        if raw_response:
            return response

        # By default, parse the response for the user.
        content_type = response.headers.get("Content-Type") or ""
        if content_type.startswith("image"):
            if not is_vision_available():
                raise ImportError(
                    f"Task '{self.task}' returned as image but Pillow is not installed."
                    " Please install it (`pip install Pillow`) or pass"
                    " `raw_response=True` to get the raw `Response` object and parse"
                    " the image by yourself."
                )

            from PIL import Image

            return Image.open(io.BytesIO(response.content))
        elif content_type == "application/json":
            return response.json()
        else:
            raise NotImplementedError(
                f"{content_type} output type is not implemented yet. You can pass"
                " `raw_response=True` to get the raw `Response` object and parse the"
                " output by yourself."
            )
