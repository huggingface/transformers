import copy
import importlib
import os
import random
from pathlib import Path

import requests

from .pipelines.test_pipelines_audio_classification import AudioClassificationPipelineTests
from .pipelines.test_pipelines_feature_extraction import FeatureExtractionPipelineTests
from .pipelines.test_pipelines_fill_mask import FillMaskPipelineTests
from .pipelines.test_pipelines_image_classification import ImageClassificationPipelineTests
from .pipelines.test_pipelines_text_classification import TextClassificationPipelineTests
from .pipelines.test_pipelines_text_generation import TextGenerationPipelineTests


pipeline_test_mapping = {
    "audio-classification": {"test": AudioClassificationPipelineTests},
    "feature-extraction": {"test": FeatureExtractionPipelineTests},
    "fill-mask": {"test": FillMaskPipelineTests},
    "image-classification": {"test": ImageClassificationPipelineTests},
    "text-classification": {"test": TextClassificationPipelineTests},
    "text-generation": {"test": TextGenerationPipelineTests},
}

for task, task_info in pipeline_test_mapping.items():
    test = task_info["test"]
    task_info["mapping"] = {
        "pt": getattr(test, "model_mapping", None),
        "tf": getattr(test, "tf_model_mapping", None),
    }


# Download tiny model summary (used to avoid requesting from Hub too many times)
url = "https://huggingface.co/datasets/hf-internal-testing/tiny-random-model-summary/raw/main/processor_classes.json"
tiny_model_summary = requests.get(url).json()

PATH_TO_TRANSFORMERS = os.path.join(Path(__file__).parent.parent, "src/transformers")


# Dynamically import the Transformers module to grab the attribute classes of the processor form their names.
spec = importlib.util.spec_from_file_location(
    "transformers",
    os.path.join(PATH_TO_TRANSFORMERS, "__init__.py"),
    submodule_search_locations=[PATH_TO_TRANSFORMERS],
)
transformers_module = spec.loader.load_module()


class PipelineTesterMixin:
    model_tester = None
    supported_frameworks = ["pt", "tf"]

    @property
    def framework(self):
        framework = None
        if "ModelTesterMixin" in set(x.__name__ for x in self.__class__.__bases__):
            framework = "pt"
        elif "TFModelTesterMixin" in set(x.__name__ for x in self.__class__.__bases__):
            framework = "tf"
        return framework

    @property
    def config_class(self):
        config = None
        method = getattr(self.model_tester, "get_config", None)
        if method is not None:
            config = method()
        if config is None:
            method = getattr(self.model_tester, "prepare_config_and_inputs", None)
            if method is not None:
                config = method()[0]
        return config.__class__ if config is not None else None

    def run_task_tests(self, task):
        """Run pipeline tests for a specific `task`

        Args:
            task (`str`):
                A task name. This should be a key in the mapping `pipeline_test_mapping`.
        """
        if self.framework not in self.supported_frameworks:
            self.skipTest(
                f"Test is skipped: Could not determined the framework. This should be in {self.supported_frameworks}, "
                f"but got `{self.framework}`)."
            )

        if task not in pipeline_test_mapping:
            self.skipTest(f"Test is skipped: task {task} is not in the mapping `pipeline_test_mapping`.")
        model_mapping = pipeline_test_mapping[task]["mapping"][self.framework]
        # `_LazyAutoMapping` always has length 0: we need to call `keys()` first before getting the length!
        if model_mapping is None or len(list(model_mapping.keys())) == 0:
            self.skipTest(
                f"Test is skipped: No model architecture under framework `{self.framework}` is found for the task"
                f" `{task}`."
            )

        if self.config_class is None:
            raise ValueError("self.config_class should not be `None`!")

        # Get model-specific architecture for the task.
        # If `config_class` is irrelevant to the pipeline task, we will skip the tests
        model_architectures = model_mapping.get(self.config_class, None)
        if model_architectures is None:
            self.skipTest(
                f"Test is skipped: No model architecture under framework `{self.framework}` with the configuration class "
                f"`{self.config_class.__name__}` is found for the task `{task}`."
            )

        if not isinstance(model_architectures, tuple):
            model_architectures = (model_architectures,)

        for model_architecture in model_architectures:
            model_arch_name = model_architecture.__name__

            # Get the canonical name
            for _prefix in ["Flax", "TF"]:
                if model_arch_name.startswith(_prefix):
                    model_arch_name = model_arch_name[len(_prefix) :]
                    break

            tokenizer_names = []
            processor_names = []
            if model_arch_name in tiny_model_summary:
                tokenizer_names = tiny_model_summary[model_arch_name]["tokenizer_classes"]
                processor_names = tiny_model_summary[model_arch_name]["processor_classes"]
            # Adding `None` (if empty) so we can generate tests
            tokenizer_names = [None] if len(tokenizer_names) == 0 else tokenizer_names
            processor_names = [None] if len(processor_names) == 0 else processor_names

            repo_name = f"tiny-random-{model_arch_name}"

            self.run_model_pipeline_tests(task, repo_name, model_architecture, tokenizer_names, processor_names)

    def run_model_pipeline_tests(self, task, repo_name, model_architecture, tokenizer_names, processor_names):
        """Run pipeline tests for a specific `task` with the give model class and tokenizer/processor class names

        Args:
            task (`str`):
                A task name. This should be a key in the mapping `pipeline_test_mapping`.
            repo_name (`str`):
                A model repository id on the Hub.
            model_architecture (`type`):
                A subclass of `PretrainedModel` or `PretrainedModel`.
            tokenizer_names (`List[str]`):
                A list of names of a subclasses of `PreTrainedTokenizerFast` or `PreTrainedTokenizer`.
            processor_names (`List[str]`):
                A list of names of subclasses of `BaseImageProcessor` or `FeatureExtractionMixin`.
        """
        for tokenizer_name in tokenizer_names:
            for processor_name in processor_names:
                self.run_pipeline_test(task, repo_name, model_architecture, tokenizer_name, processor_name)

    def run_pipeline_test(self, task, repo_name, model_architecture, tokenizer_name, processor_name):
        """Run pipeline tests for a specific `task` with the give model class and tokenizer/processor class name

        The model will be loaded from a model repository on the Hub.

        Args:
            task (`str`):
                A task name. This should be a key in the mapping `pipeline_test_mapping`.
            repo_name (`str`):
                A model repository id on the Hub.
            model_architecture (`type`):
                A subclass of `PretrainedModel` or `PretrainedModel`.
            tokenizer_name (`str`):
                The name of a subclass of `PreTrainedTokenizerFast` or `PreTrainedTokenizer`.
            processor_name (`str`):
                The name of a subclass of `BaseImageProcessor` or `FeatureExtractionMixin`.
        """
        repo_id = f"hf-internal-testing/{repo_name}"

        tokenizer = None
        if tokenizer_name is not None:
            tokenizer_class = getattr(transformers_module, tokenizer_name)
            tokenizer = tokenizer_class.from_pretrained(repo_id)

        processor = None
        if processor_name is not None:
            processor_class = getattr(transformers_module, processor_name)
            # If the required packages (like `Pillow`) are not installed, this will fail.
            try:
                processor = processor_class.from_pretrained(repo_id)
            except Exception:
                self.skipTest(f"Test is skipped: Could not load the processor from `{repo_id}` with `processor_name`.")

        try:
            model = model_architecture.from_pretrained(repo_id)
        except Exception:
            self.skipTest(f"Test is skipped: Could not load the model from `{repo_id}` with `{model_architecture}`.")

        # validate
        validate_test_components(self, model, tokenizer, processor)

        if hasattr(model, "eval"):
            model = model.eval()

        # Get an instance of the corresponding class `XXXPipelineTests` in order to use `get_test_pipeline` and
        # `run_pipeline_test`.
        task_test = pipeline_test_mapping[task]["test"]()

        pipeline, examples = task_test.get_test_pipeline(model, tokenizer, processor)
        if pipeline is None:
            # The test can disable itself, but it should be very marginal
            # Concerns: Wav2Vec2ForCTC without tokenizer test (FastTokenizer don't exist)
            self.skipTest("Test is skipped: Could not get the pipeline for testing.")

        task_test.run_pipeline_test(pipeline, examples)

        def run_batch_test(pipeline, examples):
            # Need to copy because `Conversation` are stateful
            if pipeline.tokenizer is not None and pipeline.tokenizer.pad_token_id is None:
                return  # No batching for this and it's OK

            # 10 examples with batch size 4 means there needs to be a unfinished batch
            # which is important for the unbatcher
            def data(n):
                for _ in range(n):
                    # Need to copy because Conversation object is mutated
                    yield copy.deepcopy(random.choice(examples))

            out = []
            for item in pipeline(data(10), batch_size=4):
                out.append(item)
            self.assertEqual(len(out), 10)

        run_batch_test(pipeline, examples)

    def test_pipeline_feature_extraction(self):
        self.run_task_tests(task="feature-extraction")

    def test_pipeline_audio_classification(self):
        self.run_task_tests(task="audio-classification")

    def test_pipeline_fill_mask(self):
        self.run_task_tests(task="fill-mask")

    def test_pipeline_image_classification(self):
        self.run_task_tests(task="image-classification")

    def test_pipeline_text_classification(self):
        self.run_task_tests(task="text-classification")

    def test_pipeline_text_generation(self):
        self.run_task_tests(task="text-generation")


def validate_test_components(test_case, model, tokenizer, processor):
    # TODO: Move this to tiny model creation script
    # head-specific (within a model type) necessary changes to the config
    # 1. for `BlenderbotForCausalLM`
    if model.__class__.__name__ == "BlenderbotForCausalLM":
        model.config.encoder_no_repeat_ngram_size = 0

    # TODO: Change the tiny model creation script: don't create models with problematic tokenizers
    # Avoid `IndexError` in embedding layers
    CONFIG_WITHOUT_VOCAB_SIZE = ["CanineConfig"]
    if tokenizer is not None:
        config_vocab_size = getattr(model.config, "vocab_size", None)
        # For CLIP-like models
        if config_vocab_size is None and hasattr(model.config, "text_config"):
            config_vocab_size = getattr(model.config.text_config, "vocab_size", None)
        if config_vocab_size is None and model.config.__class__.__name__ not in CONFIG_WITHOUT_VOCAB_SIZE:
            raise ValueError(
                "Could not determine `vocab_size` from model configuration while `tokenizer` is not `None`."
            )
        # TODO: Remove tiny models from the Hub which have problematic tokenizers (but still keep this block)
        if config_vocab_size is not None and len(tokenizer) > config_vocab_size:
            test_case.skipTest(
                f"Test is skipped: tokenizer (`{tokenizer.__class__.__name__}`) has {len(tokenizer)} tokens which is "
                f"greater than `config_vocab_size` ({config_vocab_size}). Something is wrong."
            )
