import jax
import jax.numpy as jnp
from bigbird_flax import FlaxBigBirdForNaturalQuestions
from datasets import load_from_disk

from transformers import BigBirdTokenizerFast


CATEGORY_MAPPING = {0: "null", 1: "short", 2: "long", 3: "yes", 4: "no"}
PUNCTUATION_SET_TO_EXCLUDE = set("".join(["‘", "’", "´", "`", ".", ",", "-", '"']))


def get_sub_answers(answers, begin=0, end=None):
    return [" ".join(x.split(" ")[begin:end]) for x in answers if len(x.split(" ")) > 1]


def expand_to_aliases(given_answers, make_sub_answers=False):
    if make_sub_answers:
        # if answers are longer than one word, make sure a predictions is correct if it coresponds to the complete 1: or :-1 sub word
        # *e.g.* if the correct answer contains a prefix such as "the", or "a"
        given_answers = (
            given_answers + get_sub_answers(given_answers, begin=1) + get_sub_answers(given_answers, end=-1)
        )
    answers = []
    for answer in given_answers:
        alias = answer.replace("_", " ").lower()
        alias = "".join(c if c not in PUNCTUATION_SET_TO_EXCLUDE else " " for c in alias)
        answers.append(" ".join(alias.split()).strip())
    return set(answers)


def get_best_valid_start_end_idx(start_scores, end_scores, top_k=1, max_size=100):
    best_start_scores, best_start_idx = jax.lax.top_k(start_scores, top_k)
    best_end_scores, best_end_idx = jax.lax.top_k(end_scores, top_k)

    widths = best_end_idx[:, None] - best_start_idx[None, :]
    mask = jnp.logical_or(widths < 0, widths > max_size)
    scores = (best_end_scores[:, None] + best_start_scores[None, :]) - (1e8 * mask)
    best_score = jnp.argmax(scores).item()

    return best_start_idx[best_score % top_k], best_end_idx[best_score // top_k]


def format_dataset(sample):
    question = sample["question"]["text"]
    context = sample["document"]["tokens"]["token"]
    is_html = sample["document"]["tokens"]["is_html"]
    long_answers = sample["annotations"]["long_answer"]
    short_answers = sample["annotations"]["short_answers"]

    context_string = " ".join([context[i] for i in range(len(context)) if not is_html[i]])

    # 0 - No ; 1 - Yes
    for answer in sample["annotations"]["yes_no_answer"]:
        if answer == 0 or answer == 1:
            return {
                "question": question,
                "context": context_string,
                "short": [],
                "long": [],
                "category": "no" if answer == 0 else "yes",
            }

    short_targets = []
    for s in short_answers:
        short_targets.extend(s["text"])
    short_targets = list(set(short_targets))

    long_targets = []
    for s in long_answers:
        if s["start_token"] == -1:
            continue
        answer = context[s["start_token"] : s["end_token"]]
        html = is_html[s["start_token"] : s["end_token"]]
        new_answer = " ".join([answer[i] for i in range(len(answer)) if not html[i]])
        if new_answer not in long_targets:
            long_targets.append(new_answer)

    category = "long_short" if len(short_targets + long_targets) > 0 else "null"

    return {
        "question": question,
        "context": context_string,
        "short": short_targets,
        "long": long_targets,
        "category": category,
    }


def main():
    dataset = load_from_disk("natural-questions-validation")
    dataset = dataset.map(format_dataset).remove_columns(["annotations", "document", "id"])
    print(dataset)

    short_validation_dataset = dataset.filter(lambda x: (len(x["question"]) + len(x["context"])) < 4 * 4096)
    short_validation_dataset = short_validation_dataset.filter(lambda x: x["category"] != "null")

    model_id = "vasudevgupta/flax-bigbird-natural-questions"
    model = FlaxBigBirdForNaturalQuestions.from_pretrained(model_id)
    tokenizer = BigBirdTokenizerFast.from_pretrained(model_id)

    @jax.jit
    def forward(*args, **kwargs):
        start_logits, end_logits, pooled_logits = model(*args, **kwargs)
        return start_logits, end_logits, jnp.argmax(pooled_logits, axis=-1)

    def evaluate(example):
        # encode question and context so that they are separated by a tokenizer.sep_token and cut at max_length
        inputs = tokenizer(
            example["question"],
            example["context"],
            return_tensors="np",
            max_length=4096,
            padding="max_length",
            truncation=True,
        )

        start_scores, end_scores, category = forward(**inputs)

        predicted_category = CATEGORY_MAPPING[category.item()]

        example["targets"] = example["long"] + example["short"]
        if example["category"] in ["yes", "no", "null"]:
            example["targets"] = [example["category"]]
        example["has_tgt"] = example["category"] != "null"
        # Now target can be: "yes", "no", "null", "list of long & short answers"

        if predicted_category in ["yes", "no", "null"]:
            example["output"] = [predicted_category]
            example["match"] = example["output"] == example["targets"]
            example["has_pred"] = predicted_category != "null"
            return example

        max_size = 38 if predicted_category == "short" else 1024
        start_score, end_score = get_best_valid_start_end_idx(
            start_scores[0], end_scores[0], top_k=8, max_size=max_size
        )

        input_ids = inputs["input_ids"][0].tolist()
        example["output"] = [tokenizer.decode(input_ids[start_score : end_score + 1])]

        answers = expand_to_aliases(example["targets"], make_sub_answers=True)
        predictions = expand_to_aliases(example["output"])

        # some preprocessing to both prediction and answer
        answers = {"".join(a.split()) for a in answers}
        predictions = {"".join(p.split()) for p in predictions}
        predictions = {s for s in predictions if s not in ["``", "''", "`", "'"]}

        # if there is a common element, it's a exact match
        example["match"] = len(list(answers & predictions)) > 0
        example["has_pred"] = predicted_category != "null" and len(predictions) > 0

        return example

    short_validation_dataset = short_validation_dataset.map(evaluate)

    total = len(short_validation_dataset)
    matched = len(short_validation_dataset.filter(lambda x: x["match"] == 1))
    print("EM score:", (matched / total) * 100, "%")


if __name__ == "__main__":
    main()
