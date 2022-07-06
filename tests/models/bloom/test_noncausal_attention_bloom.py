import torch

from transformers import BloomForCausalLM, BloomForPrefixLM
from transformers import AutoTokenizer

model_path = "bigscience/bloom-350m"
model_causal = BloomForCausalLM.from_pretrained(model_path)
model_prefix = BloomForPrefixLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

input_string = ["This is an input sentence.", "Alternative input sequence."]
target_string = ["This is one output sentence.", "Another different sentence."]

test_setting = {0:{0:{},1:{}}, 1:{0:{},1:{}}}
for i, _model in enumerate([model_prefix, model_causal]):
    for j, _input in enumerate(input_string):
        for k, _target in enumerate(target_string):
            tokenized_inputs = tokenizer(
                _input,
                return_tensors="pt"
                )

            tokenized_targets = tokenizer(
                _target,
                return_tensors="pt"
                )

            model_inputs = {}
            for key in tokenized_inputs.keys():
                model_inputs[key] = torch.concat(
                    (tokenized_inputs[key], tokenized_targets[key]),
                    -1)

            prefix_length = tokenized_inputs["input_ids"].shape[1]
            if i == 0:
                model_inputs["prefix_length"] = prefix_length
            output = _model(**model_inputs, output_attentions=True)

            test_setting[i][j][k] = {
                       "logits": output.logits,
                "prefix_length": prefix_length,
                 "input_string": _input,
                "target_string": _target,
            }

print("Test 1: Changing the inputs should change the input-side and target-side logits")
# Different input, same target
output_logits_a = test_setting[0][1][0]["logits"]
prefix_length_a = test_setting[0][1][0]["prefix_length"]

output_logits_b = test_setting[0][0][0]["logits"]
prefix_length_b = test_setting[0][0][0]["prefix_length"]

a_input_side = output_logits_a[:,:prefix_length_a, :]
a_target_side = output_logits_a[:,prefix_length_a:, :]

b_input_side = output_logits_b[:,:prefix_length_b, :]
b_target_side = output_logits_b[:,prefix_length_b:, :]

print("Input Side are the same (Should be False):", torch.equal(a_input_side,b_input_side))
print("Target Side are the same (Should be False):", torch.equal(a_target_side,b_target_side))

print("Test 2: Changing the targets should not change the input-side logits but change the target-side logits")
# Same input, difference target
output_logits_a = test_setting[0][0][0]["logits"]
prefix_length_a = test_setting[0][0][0]["prefix_length"]

output_logits_b = test_setting[0][0][1]["logits"]
prefix_length_b = test_setting[0][0][1]["prefix_length"]

a_input_side = output_logits_a[:,:prefix_length_a, :]
a_target_side = output_logits_a[:,prefix_length_a:, :]

b_input_side = output_logits_b[:,:prefix_length_b, :]
b_target_side = output_logits_b[:,prefix_length_b:, :]

print("Input Side are the same (Should be True):", torch.equal(a_input_side,b_input_side))
print("Target Side are the same (Should be False):", torch.equal(a_target_side,b_target_side))

print("Test 3: Between causal and non-causal, input-side and target-side logits should be different")
output_logits_a = test_setting[0][0][0]["logits"]
prefix_length_a = test_setting[0][0][0]["prefix_length"]

output_logits_b = test_setting[1][0][0]["logits"]
prefix_length_b = test_setting[1][0][0]["prefix_length"]

a_input_side = output_logits_a[:,:prefix_length_a, :]
a_target_side = output_logits_a[:,prefix_length_a:, :]

b_input_side = output_logits_b[:,:prefix_length_b, :]
b_target_side = output_logits_b[:,prefix_length_b:, :]

print("Input Side are the same (Should be False):", torch.equal(a_input_side,b_input_side))
print("Target Side are the same (Should be False):", torch.equal(a_target_side,b_target_side))