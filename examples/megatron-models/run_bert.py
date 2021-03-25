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
import os
import torch
from   transformers import BertTokenizer
from   transformers import MegatronBertConfig
from   transformers import MegatronBertForMaskedLM, MegatronBertForNextSentencePrediction

####################################################################################################

def main():

    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('--masked-lm', action='store_true')
    parser.add_argument('checkpoint', type=str,
        help="""Path to the folder containing the config.json and checkpoint.pt extracted from the
        NGC checkpoint using the convert_megatron_bert_checkpoint.py script.""")
    args = parser.parse_args()

    # Do we use the cased/uncased model.
    is_uncased = 'uncased' in args.checkpoint

    # The base model.
    bert = 'bert-base-' + ('uncased' if is_uncased else 'cased')
    # The tokenizer. Megatron was trained with standard tokenizer(s).
    tokenizer = BertTokenizer.from_pretrained(bert)

    # The config file.
    config_file = os.path.join(args.checkpoint, 'config.json')
    # Load the config.
    config = MegatronBertConfig.from_pretrained(config_file)
    # Make sure we do not try to tie embeddings.
    config.tie_word_embeddings = False

    # The model class.
    model_cls = MegatronBertForMaskedLM if args.masked_lm else MegatronBertForNextSentencePrediction
    # The checkpoint file.
    checkpoint_file = os.path.join(args.checkpoint, 'checkpoint.pt')
    # Load the model from transformers.
    model = model_cls.from_pretrained(checkpoint_file, config=config)

    # Do not run backward.
    model.eval()

    # Copy to the device and use FP16.
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    model.to(device)
    model.half()

    # The input sentence.

    # Create a dummy sentence (from the BERT example page).
    if args.masked_lm:
        input = tokenizer('the capital of france is [MASK]', return_tensors='pt')
        input = input.to(device)
        label = tokenizer('the capital of france is paris', return_tensors='pt')['input_ids']
        label = label.to(device)
        output = model(**input, labels=label)
    else:
        prompt = 'In Italy, pizza served in formal settings is presented unsliced.'
        next_sentence = 'The sky is blue due to the shorter wavelength of blue light.'
        input = tokenizer(prompt, next_sentence, return_tensors='pt')
        input = input.to(device)
        label = torch.LongTensor([1])
        label = label.to(device)

    # Run the model.
    output = model(**input, labels=label)

    # Outputs.
    print('loss:   ', output.loss)
    print('logits: ', output.logits)

####################################################################################################

if __name__ == '__main__':
    main()

####################################################################################################

