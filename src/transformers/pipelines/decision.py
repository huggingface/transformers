from pathlib import Path
from typing import Any

from ..decision_utils import DecisionConfig, DecisionTemplate
from ..utils import add_end_docstrings, cached_file, is_torch_available
from .base import Pipeline, build_pipeline_init_args


if is_torch_available():
    from ..modeling_decision_heads import (
        DECISION_HEAD_FOLDER,
        CalibratedTemperatureProcessor,
        LabelHead,
        NoulBandProcessor,
        TemperatureProcessor,
    )
    from ..models.auto import AutoModel


@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))
class DecisionPipeline(Pipeline):
    """
    Decision pipeline: answers typed questions (`choice`, `score`, `noul`) about a `state` with a probability for every
    allowed answer, read from one forward pass per prompt instead of generated (the System One API of Jev-like models).

    The model's `decision_config.json` (see [`DecisionConfig`]) holds its chat template, temperatures and head. The
    template marks the positions the head reads, and the head (`decision_head.safetensors`) scores the options from the
    hidden states there. Without one, a generic prompt goes through the tokenizer's chat template and the options are
    scored by the next-token logits of their labels, so any chat model can answer.

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

    def __init__(self, *args, decision_config: DecisionConfig | str | None = None, processors=None, **kwargs):
        super().__init__(*args, **kwargs)
        # The decision config and head live next to the model, or in their own repo (`decision_config="org/repo"`)
        decision_repo = decision_config if isinstance(decision_config, (str, Path)) else self.model.name_or_path
        if not isinstance(decision_config, DecisionConfig):
            try:
                decision_config = DecisionConfig.from_pretrained(decision_repo)
            except OSError:
                decision_config = DecisionConfig()
        self.decision_config = decision_config
        if decision_config.adapter is not None:
            # `subfolder` is only honored for Hub repos
            path = Path(self.model.name_or_path)
            if path.is_dir():
                self.model.load_adapter(str(path / decision_config.adapter))
            else:
                self.model.load_adapter(str(path), adapter_kwargs={"subfolder": decision_config.adapter})
        self.template = DecisionTemplate(self.tokenizer, decision_config)
        head_config = cached_file(
            decision_repo, "config.json", subfolder=DECISION_HEAD_FOLDER, _raise_exceptions_for_missing_entries=False
        )
        if head_config is None:
            self.head = LabelHead(self.tokenizer)
        else:
            self.head = AutoModel.from_pretrained(decision_repo, subfolder=DECISION_HEAD_FOLDER).to(self.device).eval()
        # Weightless post-processing of the scores, like the logits processors of `generate`
        self.processors = processors or [TemperatureProcessor(decision_config.temperature)]
        if processors is None and decision_config.calibration:
            self.processors.append(CalibratedTemperatureProcessor(**decision_config.calibration))
        if processors is None and decision_config.noul_band:
            self.processors.append(NoulBandProcessor(*decision_config.noul_band))

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def __call__(self, inputs: dict[str, Any] | list[dict[str, Any]], **kwargs):
        """
        Answer the questions of one or several System One requests.

        Args:
            inputs (`dict` or `list[dict]`):
                `{"state": ..., "questions": {name: {"type": ..., "instructions": ..., "criteria": ...}}}`, where
                `type` is `choice` (`criteria` maps each option to its description), `score` (`criteria` lists the
                levels, lowest first) or `noul` (optional `criteria` describe `true` and `false`).

        Return:
            `dict` (or a list of `dict`): The answers keyed by question name: `choice` answers have the `choice`, the
            `probabilities` of every option and a `confidence`, `score` answers the expected `score`, the `legend`, the
            `probabilities` and a `confidence`, and `noul` answers the probability `noul` of `true`.
        """
        return super().__call__(inputs, **kwargs)

    def preprocess(self, inputs):
        if not isinstance(inputs, (dict, str)):
            raise ValueError("A decision request is a dict (or its JSON) with a `state` and its `questions`.")
        return {"prompts": self.template(inputs)}

    def _forward(self, model_inputs):
        # ponytail: one forward per prompt, batch them when throughput matters
        scores = []
        for prompt in model_inputs["prompts"]:
            scores += [
                (question, s, prompt) for question, s in zip(prompt.questions, self.head.decide(self.model, prompt))
            ]
        return {"scores": scores}

    def postprocess(self, model_outputs):
        answers = {}
        for question, scores, prompt in model_outputs["scores"]:
            scores = scores.float()
            for processor in self.processors:
                scores = processor(scores, question, prompt)
            answers[question.name] = question.answer(scores.softmax(-1).tolist())
        return answers
