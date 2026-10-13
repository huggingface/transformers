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

    def __init__(self, *args, decision_config: DecisionConfig | str | None = None, processors=None, **kwargs):
        super().__init__(*args, **kwargs)
        repo = decision_config if isinstance(decision_config, (str, Path)) else self.model.name_or_path
        if not isinstance(decision_config, DecisionConfig):
            try:
                decision_config = DecisionConfig.from_pretrained(repo)
            except OSError:
                decision_config = DecisionConfig()
        self.decision_config, self.template = decision_config, DecisionTemplate(self.tokenizer, decision_config)
        if cached_file(
            repo, "config.json", subfolder=DECISION_HEAD_FOLDER, _raise_exceptions_for_missing_entries=False
        ):
            self.head = AutoModel.from_pretrained(repo, subfolder=DECISION_HEAD_FOLDER).to(self.device).eval()
        else:
            self.head = LabelHead(self.tokenizer)
        # Weightless post-processing of the scores, like the logits processors of `generate`
        self.processors = processors or [
            TemperatureProcessor(decision_config.temperature),
            *([CalibratedTemperatureProcessor(**decision_config.calibration)] if decision_config.calibration else []),
            *([NoulBandProcessor(*decision_config.noul_band)] if decision_config.noul_band else []),
        ]

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
        prompts = model_inputs["prompts"]
        return {"scores": [(q, s, p) for p in prompts for q, s in zip(p.questions, self.head.decide(self.model, p))]}

    def postprocess(self, model_outputs):
        answers = {}
        for question, scores, prompt in model_outputs["scores"]:
            scores = scores.float()
            for processor in self.processors:
                scores = processor(scores, question, prompt)
            answers[question.name] = question.answer(scores.softmax(-1).tolist())
        return answers
