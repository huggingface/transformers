import functools
import json
from pathlib import Path
from typing import Any

from ..decision import DecisionConfig, decode_decision, label_token_ids
from ..utils import add_end_docstrings, is_torch_available
from .base import Pipeline, build_pipeline_init_args


if is_torch_available():
    from ..decision.modeling_utils import DecisionMixin, read_marks


@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))
class DecisionPipeline(Pipeline):
    """
    Decision pipeline: answers typed questions (`choice`, `score`, `noul`) about a `state` with a probability for every
    option, from one forward pass per prompt (the System One API of Jev-like models). The decision config (see
    [`DecisionConfig`]) lives next to the model or in the repo given as `decision_config`. A decision model
    (`XxxForDecision`) scores the options itself; any other model scores them by the next-token logits of their labels, so any chat model can answer.

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
    >>> decide("Where is my parcel?", questions={"refund": {"type": "noul"}})["refund"]["type"]
    'noul'
    ```

    Call arguments:
        inputs (`dict`, `str` or `list`):
            `{"state": ..., "questions": {name: {"type": ..., "instructions": ..., "criteria": ...}}}` or its JSON, with
            `type` one of `choice` (`criteria` maps options to descriptions), `score` (`criteria` lists the levels,
            lowest first) or `noul` (optional `criteria` describe `true` and `false`). Requests can carry `assets`, as
            in the System One API.
        questions (`dict`, *optional*):
            Questions to ask the input, as the `state` of a request.
        assets (`dict`, *optional*):
            Images keyed by a placeholder: wherever the placeholder appears in a string of the state, the instructions
            or the criteria, the model sees the image. An image is a System One asset (`{"type": "image", "format":
            "png", "data_base64": ...}`), a PIL image, a path or a URL. Needs a model with a processor that handles
            images.

    Returns `dict` (or a list of `dict`): the answers keyed by question name, in the System One format.
    """

    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = True

    def __init__(
        self,
        *args,
        decision_config: DecisionConfig | str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        repo = decision_config if isinstance(decision_config, (str, Path)) else self.model.name_or_path
        # A decision model (`XxxForDecision`) scores the options itself, any other model by the logits of their labels
        self.is_decision_model = isinstance(self.model, DecisionMixin)
        if isinstance(decision_config, DecisionConfig):
            self.decision_config = decision_config
        elif self.is_decision_model and decision_config is None:
            self.decision_config = self.model.decision_config
        else:
            try:
                self.decision_config = DecisionConfig.from_pretrained(repo)
            except OSError:
                self.decision_config = DecisionConfig()
        if self.decision_config.adapter:  # a LoRA in a subfolder of the repo
            self.model.load_adapter(repo, adapter_kwargs={"subfolder": self.decision_config.adapter})
        self.label_token_ids = functools.cache(functools.partial(label_token_ids, self.tokenizer))

    def _sanitize_parameters(self, questions: dict[str, Any] | None = None, assets: dict[str, Any] | None = None):
        preprocess_params = {key: value for key, value in (("questions", questions), ("assets", assets)) if value}
        return preprocess_params, {}, {}

    def preprocess(self, inputs, questions: dict[str, Any] | None = None, assets: dict[str, Any] | None = None):
        request = inputs if questions is None else {"state": inputs, "questions": questions}
        if assets:
            request = (json.loads(request) if isinstance(request, str) else request) | {"assets": assets}
        processor = self.processor or self.tokenizer
        prompts = []
        for kwargs in self.decision_config.chat_template_kwargs(request, processor):
            questions = kwargs.pop("decision_questions")
            inputs = processor.apply_chat_template(**kwargs, tokenize=True, return_dict=True, return_tensors="pt")
            prompts.append((inputs, questions))
        return {"prompts": prompts}

    def _forward(self, model_inputs):
        # TODO: batch / continious batching
        scores = []
        for inputs, questions in model_inputs["prompts"]:
            if self.is_decision_model:
                scores.append(
                    self.model.decide(**inputs, decision_questions=questions, decision_config=self.decision_config)
                )
                continue
            # the logits of the labels before each mark
            inputs, marks = read_marks(inputs, self.decision_config._sentinel_ids)
            logits = self.model(**inputs, use_cache=False, logits_to_keep=marks - 1).logits[0]
            tail = inputs["input_ids"][0, int(marks[-1]) :]
            scores.append(
                [
                    self.decision_config.get_logits_processor(question.type)(
                        tail, logits[index, self.label_token_ids(tuple(question.labels))].float()
                    )
                    for index, question in enumerate(questions)
                ]
            )
        return {"prompts": model_inputs["prompts"], "scores": scores}

    def postprocess(self, model_outputs):
        answers = {}
        for (_, questions), scores in zip(model_outputs["prompts"], model_outputs["scores"]):
            answers |= decode_decision(scores, questions)
        return answers
