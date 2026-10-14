from pathlib import Path
from typing import Any

from ..decision import DecisionConfig, DecisionTemplate, label_token_ids
from ..utils import add_end_docstrings, cached_file, is_torch_available
from .base import Pipeline, build_pipeline_init_args


if is_torch_available():
    import torch

    from ..decision import CalibratedTemperatureProcessor, NoulBandProcessor, TemperatureProcessor
    from ..models.auto import AutoModel


@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))
class DecisionPipeline(Pipeline):
    """
    Decision pipeline: answers typed questions (`choice`, `score`, `noul`) about a `state` with a probability for every
    option, from one forward pass per prompt (the System One API of Jev-like models). The decision config (see
    [`DecisionConfig`]) and the head (`decision_head/`) live next to the model or in the repo given as `decision_config`;
    without them, the options are scored by the next-token logits of their labels, so any chat model can answer.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> decide = pipeline("decision", model="Qwen/Qwen2.5-0.5B-Instruct")
    >>> answers = decide(
    ...     {
    ...         "state": "I was charged twice for my order and want a refund.",
    ...         "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
    ...     }
    ... )
    >>> answers["refund"]["type"]
    'noul'
    ```
    """

    _load_processor = False
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = True

    def __init__(self, *args, decision_config: DecisionConfig | str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        repo = decision_config if isinstance(decision_config, (str, Path)) else self.model.name_or_path
        if not isinstance(decision_config, DecisionConfig):
            try:
                decision_config = DecisionConfig.from_pretrained(repo)
            except OSError:
                decision_config = DecisionConfig()
        self.template = DecisionTemplate(self.tokenizer, decision_config)
        self.head = None
        if repo and cached_file(
            repo, "config.json", subfolder="decision_head", _raise_exceptions_for_missing_entries=False
        ):
            self.head = AutoModel.from_pretrained(repo, subfolder="decision_head").to(self.device).eval()
        self.processors = [TemperatureProcessor(decision_config.temperature)]
        if decision_config.calibration:
            self.processors.append(CalibratedTemperatureProcessor(**decision_config.calibration))
        if decision_config.noul_band:
            self.processors.append(NoulBandProcessor(*decision_config.noul_band))

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def __call__(self, inputs: dict[str, Any] | str | list, **kwargs):
        """
        Answer the questions of one or several System One requests.

        Args:
            inputs (`dict`, `str` or `list`):
                `{"state": ..., "questions": {name: {"type": ..., "instructions": ..., "criteria": ...}}}` or its JSON,
                with `type` one of `choice` (`criteria` maps options to descriptions), `score` (`criteria` lists the
                levels, lowest first) or `noul` (optional `criteria` describe `true` and `false`).

        Return:
            `dict` (or a list of `dict`): The answers keyed by question name, in the System One format.
        """
        return super().__call__(inputs, **kwargs)

    def preprocess(self, inputs):
        return {"prompts": self.template(inputs)}

    def _forward(self, model_inputs):
        # ponytail: one forward per prompt, batch them when throughput matters
        outputs = []
        for prompt in model_inputs["prompts"]:
            input_ids = torch.tensor([prompt.input_ids], device=self.model.device)
            if self.head is None:  # the logits of the labels before each mark
                positions = torch.tensor(prompt.marks, device=self.model.device) - 1
                logits = self.model(input_ids=input_ids, use_cache=False, logits_to_keep=positions).logits[0]
                labels = [label_token_ids(self.tokenizer, question) for question in prompt.questions]
                scores = [logits[index, label_ids] for index, label_ids in enumerate(labels)]
            else:
                inputs = {"input_ids": input_ids, "use_cache": False}
                if hasattr(self.head, "backbone_inputs"):  # heads that change what the backbone sees, like von's
                    inputs = self.head.backbone_inputs(prompt, self.model)
                hidden_states = self.model.base_model(**inputs).last_hidden_state[0].to(self.head.dtype)
                scores = self.head(hidden_states, prompt, self.model.get_output_embeddings())
            outputs.append((prompt, scores))
        return {"scores": outputs}

    def postprocess(self, model_outputs):
        answers = {}
        for prompt, prompt_scores in model_outputs["scores"]:
            for question, scores in zip(prompt.questions, prompt_scores):
                scores = scores.float()
                for processor in self.processors:
                    scores = processor(scores, question, prompt)
                answers[question.name] = question.answer(scores.softmax(-1).tolist())
        return answers
