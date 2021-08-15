# coding=utf-8
# Copyright sayef and The HuggingFace Inc. team. All rights reserved.
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
"""Tokenization classes for FSNER."""
from ...utils import logging
from ..bert.tokenization_bert import BertTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "sayef/fsner-bert-base-uncased": "https://huggingface.co/sayef/fsner-bert-base-uncased/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "sayef/fsner-bert-base-uncased": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "sayef/fsner-bert-base-uncased": {"do_lower_case": False},
}


class FSNERTokenizer(BertTokenizer):
    r"""
    Construct a FSNER tokenizer.

    :class:`~transformers.FSNERTokenizer` is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
        )
        
        self.start_token_id, self.end_token_id = 30522, 30523 
        
    def extract_entity_from_scores(self, query, W_query, p_start, p_end, thresh=0.70):
        """
        Extracts `topk` entities from query and scores.

        Args:
            query (`List[str]`): 
                List of query strings.
            W_query (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of query sequence tokens in the vocabulary.
            p_start (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
                Scores of each token as being start token of an entity

            p_end (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`):
                Scores of each token as being end token of an entity

            topk (`int`):
                Specifies number of outputs to return after retrieving final scores.


        Returns:
            A list of lists of tuples(decoded entity, score)

        Example::

            >>> from transformers import FSNERModel, FSNERTokenizer
            >>> import torch

            >>> tokenizer = FSNERTokenizer.from_pretrained("sayef/fsner-bert-base-uncased")
            >>> model = FSNERModel.from_pretrained("sayef/fsner-bert-base-uncased")

            >>> query = ['The KWE 4000 can reach with a maximum speed from up to 450 P/min an accuracy from 50 mg']
            >>> supports = [
            ...        [
            ...            'Horizontal flow wrapper [E] Pack 403 [/E] features the new retrofit-kit „paper-ON-form“', 
            ...            '[E] Paloma Pick-and-Place-Roboter [/E] arranges the bakery products for the downstream tray-forming equipment', 
            ...            'Finally, the new [E] Kliklok ACE [/E] carton former forms cartons and trays without the use of glue', 
            ...            'We set up our pilot plant with the right [E] FibreForm® [/E] configuration to make prototypes for your marketing tests and package validation', 
            ...            'The [E] Sigpack HML [/E] is a compact horizontal flow wrapping machine. It is suited for highly reliable hermetic packaging.', 
            ...            'The [E] CAR-T5 [/E] is a reliable, purely mechanically driven cartoning machine for versatile application fields'
            ...        ]
            ...    ]
            >>> def tokenize(x):
            ...     return tokenizer(x, padding='max_length', max_length=384, truncation=True, return_tensors="pt", return_offsets_mapping=True)
            >>> W_query = tokenize(query)
            >>> W_supports = tokenize([s for support in supports for s in support])
            >>> start_scores, end_scores = model.get_start_end_token_scores(W_query, W_supports)
            >>> output = tokenizer.decode_srart_end_token_scores(query, W_query, start_scores, end_scores, thresh=0.70)
            >>> print(output)

        """

        final_outputs = []
        for idx in range(len(W_query["input_ids"])):
            start_indexes = end_indexes = range(p_start.shape[1])

            output = []
            for start_id in start_indexes:
                for end_id in end_indexes:
                    if start_id < end_id:
                        output.append((start_id, end_id, p_start[idx][start_id].item(), p_end[idx][end_id].item()))
                        
            
            output.sort(key=lambda tup: (tup[2]*tup[3]), reverse=True)
            temp = []
            for k in range(len(output)):
                if output[k][2] * output[k][3] >= thresh:
                    c_start_pos, c_end_pos = output[k][0], output[k][1]
                    decoded = self.decode(W_query['input_ids'][idx][c_start_pos:c_end_pos])
                    temp.append((decoded, output[k][2] * output[k][3]))

            final_outputs.append(temp)
            
        return final_outputs

    
