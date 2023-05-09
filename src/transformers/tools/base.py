import base64
import importlib
import inspect
import io
import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import CommitOperationAdd, HfFolder, create_commit, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, get_session

from ..dynamic_module_utils import custom_object_save, get_class_from_dynamic_module, get_imports
from ..image_utils import is_pil_image
from ..models.auto import AutoProcessor
from ..utils import (
    CONFIG_NAME,
    cached_file,
    is_accelerate_available,
    is_torch_available,
    is_vision_available,
    logging,
)


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate.utils import send_to_device


TOOL_CONFIG_FILE = "tool_config.json"


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


# docstyle-ignore
APP_FILE_TEMPLATE = """from transformers import launch_gradio_demo
from {module_name} import {class_name}

launch_gradio_demo({class_name})
"""


class Tool:
    """
    A base class for the functions used by the agent. Subclass this and implement the `__call__` method as well as the
    following class attributes:

    - **description** (`str`) -- A short description of what your tool does, the inputs it expects and the output(s) it
      wil return. For instance 'This is a tool that downloads a file from a `url`. It takes the `url` as input, and
      returns the text contained in the file'.
    - **name** (`str`) -- A performative name that will be used for your tool in the prompt to the agent. For instance
      `"text-classifier"` or `"image_generator"`.
    - **inputs** (`List[str]`) -- The list of modalities expected for the inputs (in the same order as in the call).
      Modalitiies should be `"text"`, `"image"` or `"audio"`. This is only used by `launch_gradio_demo` or to make a
      nice space from your tool.
    - **outputs** (`List[str]`) -- The list of modalities returned but the tool (in the same order as the return of the
      call method). Modalitiies should be `"text"`, `"image"` or `"audio"`. This is only used by `launch_gradio_demo`
      or to make a nice space from your tool.

    You can also override the method [`~Tool.setup`] if your tool as an expensive operation to perform before being
    usable (such as loading a model). [`~Tool.setup`] will be called the first time you use your tool, but not at
    instantiation.
    """

    description: str = "This is a tool that ..."
    name: str = ""

    inputs: List[str]
    outputs: List[str]

    def __init__(self, *args, **kwargs):
        self.is_initialized = False
        pass

    def __call__(self, *args, **kwargs):
        return NotImplemented("Write this method in your subclass of `Tool`.")

    def setup(self):
        """
        Overwrite this method here any operation that is expensive and needs to be executed before you start using your
        tool. Such as loading a big model.
        """
        self.is_initialized = True

    def save(self, output_dir):
        """
        Saves the relevant code files for your tool so it can be pushed to the Hub. This will copy the code of your
        tool in `output_dir` as well as autogenerate:

        - a config file named `tool_config.json`
        - an `app.py` file so that your tool can be converted to a space
        - a `requirements.txt` containing the names of the module used by your tool (as detected when inspecting its
          code)

        You should only use this method to save tools that are defined in a separate module (not `__main__`).

        Args:
            output_dir (`str`): The folder in which you want to save your tool.
        """
        os.makedirs(output_dir, exist_ok=True)
        # Save module file
        if self.__module__ == "__main__":
            raise ValueError(
                f"We can't save the code defining {self} in {output_dir} as it's been defined in __main__. You "
                "have to put this code in a separate module so we can include it in the saved folder."
            )
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

        tool_config = {"tool_class": full_name, "description": self.description, "name": self.name}
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
    def from_hub(cls, repo_id, model_repo_id=None, token=None, remote=False, **kwargs):
        """
        Loads a tool defined on the Hub.

        Args:
            repo_id (`str`):
                The name of the repo on the Hub where your tool is defined.
            model_repo_id (`str`, *optional*):
                If your tool uses a model and you want to use a different model than the default, you can pass a second
                repo ID or an endpoint url to this argument.
            token (`str`, *optional*):
                The token to identify you on hf.co. If unset, will use the token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            remote (`bool`, *optional*, defaults to `False`):
                Whether to use your tool by downloading the model or (if it is available) with an inference endpoint.
            kwargs:
                Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
                `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the
                others will be passed along to its init.
        """
        if remote and model_repo_id is None:
            endpoints = get_default_endpoints()
            if repo_id not in endpoints:
                raise ValueError(
                    f"Could not infer a default endpoint for {repo_id}, you need to pass one using the "
                    "`model_repo_id` argument."
                )
            model_repo_id = endpoints[repo_id]
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
            if "custom_tool" not in config:
                raise EnvironmentError(
                    f"{repo_id} does not provide a mapping to custom tools in its configuration `config.json`."
                )
            custom_tool = config["custom_tool"]
        else:
            custom_tool = config

        tool_class = custom_tool["tool_class"]
        tool_class = get_class_from_dynamic_module(tool_class, repo_id, use_auth_token=token, **hub_kwargs)

        if remote:
            return RemoteTool(model_repo_id, token=token, tool_class=tool_class)
        return tool_class(model_repo_id, token=token, **kwargs)

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload tool",
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
    ) -> str:
        """
        Upload the tool to the Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your tool to. It should contain your organization name when
                pushing to a given organization.
            commit_message (`str`, *optional*, defaults to `"Upload tool"`):
                Message to commit while pushing.
            private (`bool`, *optional*):
                Whether or not the repository created should be private.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
        """
        repo_url = create_repo(
            repo_id=repo_id, token=token, private=private, exist_ok=True, repo_type="space", space_sdk="gradio"
        )
        repo_id = repo_url.repo_id

        with tempfile.TemporaryDirectory() as work_dir:
            # Save all files.
            self.save(work_dir)
            os.listdir(work_dir)
            operations = [
                CommitOperationAdd(path_or_fileobj=os.path.join(work_dir, f), path_in_repo=f)
                for f in os.listdir(work_dir)
            ]
            logger.info(f"Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}")
            return create_commit(
                repo_id=repo_id,
                operations=operations,
                commit_message=commit_message,
                token=token,
                create_pr=create_pr,
                repo_type="space",
            )

    @staticmethod
    def from_gradio(gradio_tool):
        """
        Creates a [`Tool`] from a gradio tool.
        """

        class GradioToolWrapper(Tool):
            def __init__(self, _gradio_tool):
                super().__init__()
                self.name = _gradio_tool.name
                self.description = _gradio_tool.description

        GradioToolWrapper.__call__ = gradio_tool.run
        return GradioToolWrapper(gradio_tool)


class RemoteTool(Tool):
    """
    A [`Tool`] that will make requests to an inference endpoint.

    Args:
        endpoint_url (`str`):
            The url of the endpoint to use.
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when
            running `huggingface-cli login` (stored in `~/.huggingface`).
        tool_class (`type`, *optional*):
            The corresponding `tool_class` if this is a remote version of an existing tool. Will help determine when
            the output should be converted to another type (like images).
    """

    def __init__(self, endpoint_url=None, token=None, tool_class=None):
        self.endpoint_url = endpoint_url
        self.client = EndpointClient(endpoint_url, token=token)
        self.tool_class = tool_class

    def prepare_inputs(self, *args, **kwargs):
        """
        Prepare the inputs received for the HTTP client sending data to the endpoint. Positional arguments will be
        matched with the signature of the `tool_class` if it was provided at instantation. Images will be encoded into
        bytes.

        You can override this method in your custom class of [`RemoteTool`].
        """
        inputs = kwargs.copy()
        if len(args) > 0:
            if self.tool_class is not None:
                # Match args with the signature
                if issubclass(self.tool_class, PipelineTool):
                    call_method = self.tool_class.encode
                else:
                    call_method = self.tool_class.__call__
                signature = inspect.signature(call_method).parameters
                parameters = [
                    k
                    for k, p in signature.items()
                    if p.kind not in [inspect._ParameterKind.VAR_POSITIONAL, inspect._ParameterKind.VAR_KEYWORD]
                ]
                if parameters[0] == "self":
                    parameters = parameters[1:]
                if len(args) > len(parameters):
                    raise ValueError(
                        f"{self.tool_class} only accepts {len(parameters)} arguments but {len(args)} were given."
                    )
                for arg, name in zip(args, parameters):
                    inputs[name] = arg
            elif len(args) > 1:
                raise ValueError("A `RemoteTool` can only accept one positional input.")
            elif len(args) == 1:
                if is_pil_image(args[0]):
                    return {"inputs": self.client.encode_image(args[0])}
                return {"inputs": args[0]}

        for key, value in inputs.items():
            if is_pil_image(value):
                inputs[key] = self.client.encode_image(value)

        return {"inputs": inputs}

    def extract_outputs(self, outputs):
        """
        You can override this method in your custom class of [`RemoteTool`] to apply some custom post-processing of the
        outputs of the endpoint.
        """
        return outputs

    def __call__(self, *args, **kwargs):
        output_image = self.tool_class is not None and self.tool_class.outputs == ["image"]
        inputs = self.prepare_inputs(*args, **kwargs)
        if isinstance(inputs, dict):
            outputs = self.client(**inputs, output_image=output_image)
        else:
            outputs = self.client(inputs, output_image=output_image)
        if isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], list):
            outputs = outputs[0]
        return self.extract_outputs(outputs)


class PipelineTool(Tool):
    """
    A [`Tool`] tailored towards Transformer models. On top of the class attributes of the base class [`Tool`], you will
    meed to specify:

    - **model_class** (`type`) -- The class to use to load the model in this tool.
    - **default_checkpoint** (`str`) -- The default checkpoint that should be used when the user doesn't specify one.
    - **pre_processor_class** (`type`, *optional*, defaults to [`AutoProcessor`]) -- The class to use to load the
      pre-processor
    - **post_processor_class** (`type`, *optional*, defaults to [`AutoProcessor`]) -- The class to use to load the
      post-processor (when different from the pre-processor).

    Args:
        model (`str` or [`PreTrainedModel`], *optional*):
            The name of the checkpoint to use for the model, or the instantiated model. If unset, will default to the
            value of the class attribute `default_checkpoint`.
        pre_processor (`str` or `Any`, *optional*):
            The name of the checkpoint to use for the pre-processor, or the instantiated pre-processor (can be a
            tokenizer, an image processor, a feature extractor or a processor). Will default to the value of `model` if
            unset.
        post_processor (`str` or `Any`, *optional*):
            The name of the checkpoint to use for the post-processor, or the instantiated pre-processor (can be a
            tokenizer, an image processor, a feature extractor or a processor). Will default to the `pre_processor` if
            unset.
        device (`int`, `str` or `torch.device`, *optional*):
            The device on which to execute the model. Will default to any accelerator available (GPU, MPS etc...), the
            CPU otherwise.
        device_map (`str` or `dict`, *optional*):
            If passed along, will be used to instantiate the model.
        model_kwargs (`dict`, *optional*):
            Any keyword argument to send to the model instantiation.
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when
            running `huggingface-cli login` (stored in `~/.huggingface`).
        hub_kwargs:
            Any additional keyword argument to send to the methods that will load the data from the Hub.
    """

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
        """
        Instantiates the `pre_processor`, `model` and `post_processor` if necessary.
        """
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

    def encode(self, raw_inputs):
        """
        Uses the `pre_processor` to prepare the inputs for the `model`.
        """
        return self.pre_processor(raw_inputs)

    def forward(self, inputs):
        """
        Sends the inputs through the `model`.
        """
        return self.model(**inputs)

    def decode(self, outputs):
        """
        Uses the `post_processor` to decode the model output.
        """
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
    """
    Launches a gradio demo for a tool. The corresponding tool class needs to properly implement the class attributes
    `inputs` and `outputs`.

    Args:
        tool_class (`type`): The class of the tool for which to launch the demo.
    """
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
    "document-question-answering": "DocumentQuestionAnsweringTool",
    "image-captioning": "ImageCaptioningTool",
    "image-question-answering": "ImageQuestionAnsweringTool",
    "image-segmentation": "ImageSegmentationTool",
    "speech-to-text": "SpeechToTextTool",
    "summarization": "TextSummarizationTool",
    "text-classification": "TextClassificationTool",
    "text-question-answering": "TextQuestionAnsweringTool",
    "text-to-speech": "TextToSpeechTool",
    "translation": "TranslationTool",
}


def get_default_endpoints():
    endpoints_file = cached_file("huggingface-tools/default-endpoints", "default_endpoints.json", repo_type="dataset")
    with open(endpoints_file, "r", encoding="utf-8") as f:
        endpoints = json.load(f)
    return endpoints


def supports_remote(task_or_repo_id):
    endpoints = get_default_endpoints()
    return task_or_repo_id in endpoints


def load_tool(task_or_repo_id, model_repo_id=None, remote=False, token=None, **kwargs):
    """
    Main function to quickly load a tool, be it on the Hub or in the Transformers library.

    Args:
        task_or_repo_id (`str`):
            The task for which to load the tool or a repo ID of a tool on the Hub. Tasks implemented in Transformers
            are:

            - `"document-question-answering"`
            - `"image-captioning"`
            - `"image-question-answering"`
            - `"image-segmentation"`
            - `"speech-to-text"`
            - `"summarization"`
            - `"text-classification"`
            - `"text-question-answering"`
            - `"text-to-speech"`
            - `"translation"`

        model_repo_id (`str`, *optional*):
            Use this argument to use a different model than the default one for the tool you selected.
        remote (`bool`, *optional*, defaults to `False`):
            Whether to use your tool by downloading the model or (if it is available) with an inference endpoint.
        token (`str`, *optional*):
            The token to identify you on hf.co. If unset, will use the token generated when running `huggingface-cli
            login` (stored in `~/.huggingface`).
        kwargs:
            Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
            `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the others
            will be passed along to its init.
    """
    if task_or_repo_id in TASK_MAPPING:
        tool_class_name = TASK_MAPPING[task_or_repo_id]
        main_module = importlib.import_module("transformers")
        tools_module = main_module.tools
        tool_class = getattr(tools_module, tool_class_name)

        if remote:
            if model_repo_id is None:
                endpoints = get_default_endpoints()
                if task_or_repo_id not in endpoints:
                    raise ValueError(
                        f"Could not infer a default endpoint for {task_or_repo_id}, you need to pass one using the "
                        "`model_repo_id` argument."
                    )
                model_repo_id = endpoints[task_or_repo_id]
            return RemoteTool(model_repo_id, token=token, tool_class=tool_class)
        else:
            return tool_class(model_repo_id, token=token, **kwargs)
    else:
        return Tool.from_hub(task_or_repo_id, model_repo_id=model_repo_id, token=token, remote=remote, **kwargs)


def add_description(description):
    """
    A decorator that adds a description to a function.
    """

    def inner(func):
        func.description = description
        func.name = func.__name__
        return func

    return inner


## Will move to the Hub
class EndpointClient:
    def __init__(self, endpoint_url: str, token: Optional[str] = None):
        if token is None:
            token = HfFolder().get_token()
        self.headers = {"authorization": f"Bearer {token}", "Content-Type": "application/json"}
        self.endpoint_url = endpoint_url

    @staticmethod
    def encode_image(image):
        _bytes = io.BytesIO()
        image.save(_bytes, format="PNG")
        b64 = base64.b64encode(_bytes.getvalue())
        return b64.decode("utf-8")

    @staticmethod
    def decode_image(raw_image):
        if not is_vision_available():
            raise ImportError(
                "This tool returned an image but Pillow is not installed. Please install it (`pip install Pillow`)."
            )

        from PIL import Image

        b64 = base64.b64decode(raw_image)
        _bytes = io.BytesIO(b64)
        return Image.open(_bytes)

    def __call__(
        self,
        inputs: Optional[Union[str, Dict, List[str], List[List[str]]]] = None,
        params: Optional[Dict] = None,
        data: Optional[bytes] = None,
        output_image: bool = False,
    ) -> Any:
        # Build payload
        payload = {}
        if inputs:
            payload["inputs"] = inputs
        if params:
            payload["parameters"] = params

        # Make API call
        response = get_session().post(self.endpoint_url, headers=self.headers, json=payload, data=data)

        # By default, parse the response for the user.
        if output_image:
            return self.decode_image(response.content)
        else:
            return response.json()
