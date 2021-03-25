####################################################################################################

# Copyright (c) 2021-, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

####################################################################################################

import argparse

import torch

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


####################################################################################################


def main():

    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="See examples in README.md.")
    args = parser.parse_args()

    # The tokenizer. Megatron was trained with standard tokenizer(s).
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # The config file.
    config_file = args.checkpoint + "_config.json"
    # Load the GPT2 config.
    config = GPT2Config.from_pretrained(config_file)

    # The checkpoint file.
    checkpoint_file = args.checkpoint + "_checkpoint.pt"
    # Load GPT2 model from transformers.
    model = GPT2LMHeadModel.from_pretrained(checkpoint_file, config=config)

    # Do not run backward.
    model.eval()

    # Copy to the device and use FP16.
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    model.to(device)
    model.half()

    # Create an empty sentence.
    input = tokenizer.encode("", return_tensors="pt")
    input = input.to(device)

    # The token ids.
    if input.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = input

    # Generate the sentence.
    output = model.generate(
        input_ids=input_ids,
        max_length=128,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1,
    )

    # Output the text.
    for sentence in output:
        sentence = sentence.tolist()
        text = tokenizer.decode(sentence, clean_up_tokenization_spaces=True)
        print(text)


####################################################################################################

if __name__ == "__main__":
    main()

####################################################################################################
