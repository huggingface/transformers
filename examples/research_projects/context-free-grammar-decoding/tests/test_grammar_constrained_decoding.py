from transformers_cfg.generation import GrammarConstrainedLogitsProcessor
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint

from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_IDS = [
    "hf-internal-testing/tiny-random-GPTJForCausalLM",
    "hf-internal-testing/tiny-random-BloomForCausalLM",
    "hf-internal-testing/tiny-random-PhiForCausalLM",
    "hf-internal-testing/tiny-random-gpt2",
    # "hf-internal-testing/tiny-random-BlenderbotForCausalLM",
]


def test_grammar_constrained_decoding_greedy_w_number_grammar():
    # test greedy decoding with grammar constraints
    grammar_str = """
    root ::= [0-9]+
    """

    for model_id in MODEL_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        grammar = IncrementalGrammarConstraint(grammar_str, start_rule_name="root", tokenizer=tokenizer)
        grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

        prefix = "This is a valid number:"

        input_ids = tokenizer([prefix], add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"]

        output = model.generate(
            input_ids,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1,
            max_new_tokens=40,
            top_p=0.92,
            top_k=5,
            logits_processor=[grammar_processor],
            repetition_penalty=100.0,
            early_stopping=True,
        )

        # generations = tokenizer.batch_decode(output, skip_special_tokens=True)

        generations = tokenizer.batch_decode(output[:, input_ids.shape[1] :], skip_special_tokens=True)
        assert generations[0].isdigit(), f"generations: {generations} is not a number"


def test_grammar_constrained_decoding_greedy_w_balanced_parenthesis_grammar():
    # test greedy decoding with grammar constraints
    grammar_str = """
    root ::= "(" root ")" | ""
    """

    for model_id in MODEL_IDS:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        grammar = IncrementalGrammarConstraint(grammar_str, start_rule_name="root", tokenizer=tokenizer)
        grammar_processor = GrammarConstrainedLogitsProcessor(grammar)

        prefix = "This is a valid json:"

        input_ids = tokenizer([prefix], add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"]
        MAX_NEW_TOKENS = 20

        output = model.generate(
            input_ids,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=1,
            max_new_tokens=MAX_NEW_TOKENS,
            top_p=0.92,
            top_k=5,
            logits_processor=[grammar_processor],
            repetition_penalty=100.0,
            early_stopping=True,
        )

        # generations = tokenizer.batch_decode(output, skip_special_tokens=True)

        generation: str = tokenizer.batch_decode(output[:, input_ids.shape[1] :], skip_special_tokens=True)[0]

        def check_parentheses(generation):
            stack = []
            for char in generation:
                if char == "(":
                    stack.append(char)
                elif char == ")":
                    if not stack:
                        return False
                    stack.pop()
            return not stack

        assert check_parentheses(generation), f"generations: {generation} is not a balanced parenthesis"


if __name__ == "__main__":
    test_grammar_constrained_decoding_greedy_w_number_grammar()
    test_grammar_constrained_decoding_greedy_w_balanced_parenthesis_grammar()
