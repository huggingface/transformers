# coding=utf-8
# Copyright 2020 (...) and The HuggingFace Inc. team.
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

import collections
import ast
from typing import Any, Dict, Iterable, List, Mapping, Optional, overload, Text, Tuple, Union
import dataclasses

import torch

from .tokenization_bert import BertTokenizer, BertTokenizerFast
from .tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    INIT_TOKENIZER_DOCSTRING,
    AddedToken,
    BatchEncoding,
    EncodedInput,
    EncodedInputPair,
    PaddingStrategy,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    TensorType,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)
from src.transformers import tokenization_tapas_utilities as utils

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        # to be added
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    # to be added
}


PRETRAINED_INIT_CONFIGURATION = {
    # to be added
}

# @dataclasses.dataclass(frozen=True)
# class Token:
#   original_text: Text
#   piece: Text

# def _get_pieces(tokens):
#   return (token.piece for token in tokens)


@dataclasses.dataclass(frozen=True)
class TokenCoordinates:
  column_index: int
  row_index: int
  token_index: int


@dataclasses.dataclass
class TokenizedTable:
  rows: List[List[List[Text]]]
  selected_tokens: List[TokenCoordinates]


@dataclasses.dataclass(frozen=True)
class SerializedExample:
  tokens: List[Text]
  column_ids: List[int]
  row_ids: List[int]
  segment_ids: List[int]
  

def _is_inner_wordpiece(token):
    return token.startswith('##')


class TapasTokenizer(BertTokenizer):
    r"""
    Construct an TAPAS tokenizer.

    :class:`~transformers.TapasTokenizer` inherits from :class:`~transformers.BertTokenizer` since it uses the same
    vocabulary. However, it adds several token type ids to encode tabular structure. It runs end-to-end
    tokenization on a table and associated queries: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION

    def __init__(self, 
                cell_trim_length: int = -1,
                max_column_id: int = None,
                max_row_id: int = None,
                strip_column_names: bool = False,
                # add_aggregation_candidates: bool = False, # suggestion, remove this?
                # expand_entity_descriptions: bool = False, # suggestion, remove this?
                # entity_descriptions_sentence_limit: int = 5, # suggestion, remove this?
                #use_document_title: bool = False, # suggestion: remove this?
                update_answer_coordinates: bool = False, # Re-compute answer coordinates from the answer text.
                drop_rows_to_fit: bool = False, # Drop last rows if table doesn't fit within max sequence length.
                **kwargs):
        super().__init__(**kwargs)

        # Added tokens - We store this for both slow and fast tokenizers
        # until the serialization of Fast tokenizers is updated
        self.added_tokens_encoder: Dict[str, int] = {}
        self.added_tokens_decoder: Dict[int, str] = {}
        self.unique_no_split_tokens: List[str] = []

        # Additional properties 
        self.cell_trim_length = cell_trim_length
        self.max_column_id = max_column_id if max_column_id is not None else self.model_max_length
        self.max_row_id = max_row_id if max_row_id is not None else self.model_max_length
        self.strip_column_names = strip_column_names
        # self.add_aggregation_candidates = add_aggregation_candidates, suggestion: remove this?
        # self.expand_entity_descriptions = expand_entity_descriptions, suggestion: remove this?
        # self.entity_descriptions_sentence_limit = entity_descriptions_sentence_limit, suggestion: remove this?
        #self.use_document_title = use_document_title, suggestion: remove this?
        self.update_answer_coordinates = update_answer_coordinates
        self.drop_rows_to_fit = drop_rows_to_fit
    
    def _tokenize_table(
        self,
        table = None,
        ):
        """Runs tokenizer over columns and table cell texts."""
        tokenized_rows = []
        tokenized_row = []
        # tokenize column headers
        for column in table:
            if self.strip_column_names:
                tokenized_row.append(self.tokenize(''))
            else:
                tokenized_row.append(self.tokenize(column))
        tokenized_rows.append(tokenized_row)

        # tokenize cell values
        for idx, row in table.iterrows():
            tokenized_row = []
            for cell in row:
                tokenized_row.append(self.tokenize(cell))
            tokenized_rows.append(tokenized_row)

        token_coordinates = []
        for row_index, row in enumerate(tokenized_rows):
            for column_index, cell in enumerate(row):
                for token_index, _ in enumerate(cell):
                    token_coordinates.append(
                        TokenCoordinates(
                            row_index=row_index,
                            column_index=column_index,
                            token_index=token_index,
                        ))

        return TokenizedTable(
            rows=tokenized_rows,
            selected_tokens=token_coordinates,
        )
    
    def question_encoding_cost(self, question_tokens):
        # Two extra spots of SEP and CLS.
        return len(question_tokens) + 2
    
    def _get_token_budget(self, question_tokens):
        return self.model_max_length - self.question_encoding_cost(question_tokens)
    
    def _get_table_values(self, table, num_columns, num_rows, num_tokens):
        """Iterates over partial table and returns token, col. and row indexes."""
        for tc in table.selected_tokens:
            # First row is header row.
            if tc.row_index >= num_rows + 1:
                continue
            if tc.column_index >= num_columns:
                continue
            cell = table.rows[tc.row_index][tc.column_index]
            token = cell[tc.token_index]
            word_begin_index = tc.token_index
            # Don't add partial words. Find the starting word piece and check if it
            # fits in the token budget.
            while word_begin_index >= 0 \
                and _is_inner_wordpiece(cell[word_begin_index]):
                word_begin_index -= 1
            if word_begin_index >= num_tokens:
                continue
            yield token, tc.column_index + 1, tc.row_index
    
    def _get_table_boundaries(self, table):
        """Return maximal number of rows, columns and tokens."""
        max_num_tokens = 0
        max_num_columns = 0
        max_num_rows = 0
        for tc in table.selected_tokens:
            max_num_columns = max(max_num_columns, tc.column_index + 1)
            max_num_rows = max(max_num_rows, tc.row_index + 1)
            max_num_tokens = max(max_num_tokens, tc.token_index + 1)
            max_num_columns = min(self.max_column_id, max_num_columns)
            max_num_rows = min(self.max_row_id, max_num_rows)
        return max_num_rows, max_num_columns, max_num_tokens

    def _get_table_cost(self, table, num_columns,
                        num_rows, num_tokens):
        return sum(1 for _ in self._get_table_values(table, num_columns, num_rows,
                                                    num_tokens))
    
    def _get_max_num_tokens(
        self,
        question_tokens,
        tokenized_table,
        num_columns,
        num_rows,
    ):
        """Computes max number of tokens that can be squeezed into the budget."""
        token_budget = self._get_token_budget(question_tokens)
        _, _, max_num_tokens = self._get_table_boundaries(tokenized_table)
        if self.cell_trim_length >= 0 and max_num_tokens > self.cell_trim_length:
            max_num_tokens = self.cell_trim_length
        num_tokens = 0
        for num_tokens in range(max_num_tokens + 1):
            cost = self._get_table_cost(tokenized_table, num_columns, num_rows,
                                        num_tokens + 1)
            if cost > token_budget:
                break
        if num_tokens < max_num_tokens:
            if self.cell_trim_length >= 0:
                # We don't allow dynamic trimming if a cell_trim_length is set.
                return None
            if num_tokens == 0:
                return None
        return num_tokens
    
    def _get_num_columns(self, table):
        num_columns = table.shape[1]
        if num_columns >= self.max_column_id:
            raise ValueError('Too many columns')
        return num_columns

    def _get_num_rows(self, table, drop_rows_to_fit):
        num_rows = table.shape[0]
        if num_rows >= self.max_row_id:
            if drop_rows_to_fit:
                num_rows = self.max_row_id - 1
            else:
                raise ValueError('Too many rows')
        return num_rows
    
    def _serialize_text(self, question_tokens):
        """Serialzes texts in index arrays."""
        tokens = []
        segment_ids = []
        column_ids = []
        row_ids = []

        # add [CLS] token at the beginning
        tokens.append(self.cls_token)
        segment_ids.append(0)
        column_ids.append(0)
        row_ids.append(0)

        for token in question_tokens:
            tokens.append(token)
            segment_ids.append(0)
            column_ids.append(0)
            row_ids.append(0)

        return tokens, segment_ids, column_ids, row_ids

    def _serialize(
        self,
        question_tokens,
        table,
        num_columns,
        num_rows,
        num_tokens,
    ):
        """Serializes table and text."""
        tokens, segment_ids, column_ids, row_ids = self._serialize_text(
            question_tokens)

        # add [SEP] token between question and table tokens
        tokens.append(self.sep_token)
        segment_ids.append(0)
        column_ids.append(0)
        row_ids.append(0)

        for token, column_id, row_id in self._get_table_values(
            table, num_columns, num_rows, num_tokens):
            tokens.append(token)
            segment_ids.append(1)
            column_ids.append(column_id)
            row_ids.append(row_id)

        return SerializedExample(
            tokens=tokens,
            segment_ids=segment_ids,
            column_ids=column_ids,
            row_ids=row_ids,
        )
        
    def _get_column_values(self, table_numeric_values):
        """This is an adaptation from _get_column_values in tf_example_utils.py.
        Given table_numeric_values, a dictionary that maps row indices of a certain column 
        of a Pandas dataframe to either an empty list (no numeric value) or a list containing 
        a NumericValue object, it returns the same dictionary, but only for the row indices that 
        have a corresponding NumericValue object. 
        """
        table_numeric_values_without_empty_lists = {}
        for row_index, value in table_numeric_values.items():
            if len(value) != 0:
                table_numeric_values_without_empty_lists[row_index] = value[0]
        return table_numeric_values_without_empty_lists
    
    def _get_cell_token_indexes(self, column_ids, row_ids, column_id, row_id):
        for index in range(len(column_ids)):
            if (column_ids[index] - 1 == column_id and row_ids[index] - 1 == row_id):
                yield index
    
    def _add_numeric_column_ranks(self, column_ids, row_ids,
                                table,
                                features):
        """Adds column ranks for all numeric columns."""

        ranks = [0] * len(column_ids)
        inv_ranks = [0] * len(column_ids)

        # here, some complex code involving functions from number_annotations_utils are used in the original implementation
        columns_to_numeric_values = {}
        if table is not None:
            for col_index in range(len(table.columns)):
                table_numeric_values = utils._parse_column_values(table, col_index)
                # we remove row indices for which no numeric value was found
                table_numeric_values = self._get_column_values(table_numeric_values)
                # we add the numeric values to a dictionary, to be used in _add_numeric_relations
                columns_to_numeric_values[col_index] = table_numeric_values
                if not table_numeric_values:
                    continue

                try:
                    key_fn = utils.get_numeric_sort_key_fn(
                        table_numeric_values.values())
                except ValueError:
                    continue

                table_numeric_values = {
                    row_index: key_fn(value)
                    for row_index, value in table_numeric_values.items()
                }

                table_numeric_values_inv = collections.defaultdict(list)
                for row_index, value in table_numeric_values.items():
                    table_numeric_values_inv[value].append(row_index)

                unique_values = sorted(table_numeric_values_inv.keys())

                for rank, value in enumerate(unique_values):
                    for row_index in table_numeric_values_inv[value]:
                        for index in self._get_cell_token_indexes(column_ids, row_ids, col_index, row_index):
                            ranks[index] = rank + 1
                            inv_ranks[index] = len(unique_values) - rank

        features['column_ranks'] = ranks
        features['inv_column_ranks'] = inv_ranks

        return features, columns_to_numeric_values

    def _get_numeric_sort_key_fn(self, table_numeric_values, value):
        """Returns the sort key function for comparing value to table values.
        The function returned will be a suitable input for the key param of the
        sort(). See number_annotation_utils._get_numeric_sort_key_fn for details.
        Args:
        table_numeric_values: Numeric values of a column
        value: Numeric value in the question.
        Returns:
        A function key function to compare column and question values.
        """
        if not table_numeric_values:
            return None
        all_values = list(table_numeric_values.values())
        all_values.append(value)
        try:
            return utils.get_numeric_sort_key_fn(all_values)
        except ValueError:
            return None
    
    def _add_numeric_relations(self, question,
                             column_ids, row_ids,
                             table,
                             features,
                             columns_to_numeric_values):
        """Adds numeric relation embeddings to 'features'.
        Args:
        question: The question, numeric values are used.
        column_ids: Maps word piece position to column id.
        row_ids: Maps word piece position to row id.
        table: The table containing the numeric cell values.
        features: Output.
        columns_to_numeric_values: Dictionary that maps column indices to numeric values.
        """

        numeric_relations = [0] * len(column_ids)

        # TO BE ADDED (see original implementation)
        # first, we add any numeric value spans to the question:
        # Create a dictionary that maps a table cell to the set of all relations
        # this cell has with any value in the question.
        cell_indices_to_relations = collections.defaultdict(set)
        if question is not None and table is not None:
            question, numeric_spans = utils.add_numeric_values_to_question(question)
            for numeric_value_span in numeric_spans:
                for value in numeric_value_span.values:
                    for column_index in range(len(table.columns)):
                        table_numeric_values = columns_to_numeric_values[column_index]
                        sort_key_fn = self._get_numeric_sort_key_fn(table_numeric_values,
                                                                value)
                        if sort_key_fn is None:
                            continue
                        for row_index, cell_value in table_numeric_values.items():
                            relation = utils.get_numeric_relation(value, cell_value, sort_key_fn)
                            if relation is not None:
                                cell_indices_to_relations[column_index, row_index].add(relation)
        
        # For each cell add a special feature for all its word pieces.
        for (column_index, row_index), relations in cell_indices_to_relations.items():
            relation_set_index = 0
            for relation in relations:
                assert relation.value >= utils.Relation.EQ.value
                relation_set_index += 2**(relation.value - utils.Relation.EQ.value)
            for cell_token_index in self._get_cell_token_indexes(column_ids, row_ids,
                                                        column_index, row_index):
                numeric_relations[cell_token_index] = relation_set_index
        
        features['numeric_relations'] = numeric_relations

        return features

    def _add_numeric_values(self, 
                          table,
                          token_ids_dict,
                          features,
                          columns_to_numeric_values):
        """Adds numeric values for computation of answer loss."""
        
        numeric_values = [float('nan')] * self.model_max_length

        if table is not None:
            num_rows = table.shape[0] 
            num_columns = table.shape[1]

            for col_index in range(num_columns):
                if not columns_to_numeric_values[col_index]:
                    continue
                else:
                    for row_index in range(num_rows):
                        numeric_value = columns_to_numeric_values[col_index][row_index]
                        if numeric_value.float_value is None:
                            continue

                        float_value = numeric_value.float_value
                        if float_value == float('inf'):
                            continue

                        for index in self._get_cell_token_indexes(token_ids_dict['column_ids'],
                                                            token_ids_dict['row_ids'],
                                                            col_index, row_index):
                            numeric_values[index] = float_value

        features['numeric_values'] = numeric_values

        return features

    def _add_numeric_values_scale(self, table, token_ids_dict, features):
        """Adds a scale to each token to down weigh the value of long words."""
        
        numeric_values_scale = [1.0] * self.model_max_length
        
        if table is None:
            return numeric_values_scale
        
        num_rows = table.shape[0]
        num_columns = table.shape[1]
        
        for col_index in range(num_columns):
            for row_index in range(num_rows):
                indices = [
                    index for index in self._get_cell_token_indexes(
                        token_ids_dict['column_ids'], token_ids_dict['row_ids'],
                        col_index, row_index)
                ]
                num_indices = len(indices)
                if num_indices > 1:
                    for index in indices:
                        numeric_values_scale[index] = float(num_indices)

        features['numeric_values_scale'] = numeric_values_scale

        return features
    
    def _pad_to_seq_length(self, inputs):
        while len(inputs) > self.model_max_length:
            inputs.pop()
        while len(inputs) < self.model_max_length:
            inputs.append(0)
    
    def _to_features(self, tokens, token_ids_dict, table, question):
        """Produces a dict of features. This function creates input ids, attention mask, token type ids
        (except the prev label ids), as well as numeric value and numeric value scale. 
        """
        tokens = list(tokens)
        token_ids_dict = {
            key: list(values) for key, values in token_ids_dict.items()
        }

        length = len(tokens)
        for values in token_ids_dict.values():
            if len(values) != length:
                raise ValueError('Inconsistent length')

        # currently the input ids, mask and token type ids are created here 
        # also, padding and truncation up to max length is done here (see function _pad_to_seq_length)
        # (later, this will be done in prepare_for_model)   

        input_ids = self.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        self._pad_to_seq_length(input_ids)
        self._pad_to_seq_length(attention_mask)
        for values in token_ids_dict.values():
            self._pad_to_seq_length(values)

        assert len(input_ids) == self.model_max_length
        assert len(attention_mask) == self.model_max_length
        for values in token_ids_dict.values():
            assert len(values) == self.model_max_length

        features = {}
        features['input_ids'] = input_ids
        features['attention_mask'] = attention_mask
        for key, values in sorted(token_ids_dict.items()):
             features[key] = values

        features, columns_to_numeric_values = self._add_numeric_column_ranks(token_ids_dict['column_ids'],
                                   token_ids_dict['row_ids'], table, features)

        features = self._add_numeric_relations(question, token_ids_dict['column_ids'],
                                    token_ids_dict['row_ids'], table, features, columns_to_numeric_values)

        # finally, add numeric values and numeric values scale (only needed in case of loss calculation)
        # so they should only be returned in case answer_coordinates + answer_texts are provided
        
        features = self._add_numeric_values(table, token_ids_dict, features, columns_to_numeric_values)

        features = self._add_numeric_values_scale(table, token_ids_dict, features)

        # we do not add table id and table id hash
        #if table:
        #    features['table_id'] = create_string_feature([table.table_id.encode('utf8')])
        #    features['table_id_hash'] = create_int_feature([fingerprint(table.table_id) % _MAX_INT])
        
        return features
    
    def _to_trimmed_features(
            self,
            question,
            table,
            question_tokens,
            tokenized_table,
            num_columns,
            num_rows,
            drop_rows_to_fit = False,
    ):
        """Finds optimal number of table tokens to include and serializes."""
        init_num_rows = num_rows
        while True:
            num_tokens = self._get_max_num_tokens(
                question_tokens,
                tokenized_table,
                num_rows=num_rows,
                num_columns=num_columns,
            )
            if num_tokens is not None:
                # We could fit the table.
                break
            if not drop_rows_to_fit or num_rows == 0:
                raise ValueError('Sequence too long')
            # Try to drop a row to fit the table.
            num_rows -= 1
        
        serialized_example = self._serialize(question_tokens, tokenized_table,
                                            num_columns, num_rows, num_tokens)

        assert len(serialized_example.tokens) <= self.model_max_length

        feature_dict = {
            'column_ids': serialized_example.column_ids,
            'row_ids': serialized_example.row_ids,
            'segment_ids': serialized_example.segment_ids,
        }

        features = self._to_features(
                serialized_example.tokens, feature_dict, table=table, question=question)

        return serialized_example, features

    #### Everything related to label ids calculation ####

    def _get_all_answer_ids_from_coordinates(
            self,
            column_ids,
            row_ids,
            answers_list,
    ):
        """Maps lists of answer coordinates to token indexes."""
        answer_ids = [0] * len(column_ids)
        found_answers = set()
        all_answers = set()
        for answers in answers_list:
            column_index, row_index = answers
            all_answers.add((column_index, row_index))
            for index in self._get_cell_token_indexes(column_ids, row_ids, column_index,
                                                row_index):
                found_answers.add((column_index, row_index))
                answer_ids[index] = 1

        missing_count = len(all_answers) - len(found_answers)
        return answer_ids, missing_count

    def _get_all_answer_ids(
        self,
        column_ids,
        row_ids,
        question,
        answer_coordinates
    ):
        """Maps lists of questions with answer coordinates to token indexes.
        Here, we swap column and row coordinates. In the TSV format, the coordinates
        are given as (row, column) tuples. Here, we swap them to (column, row) format.
        """

        def _to_coordinates(
            question, answer_coordinates_question):
            return [(coords[1], coords[0])
                    for coords in answer_coordinates_question]

        return self._get_all_answer_ids_from_coordinates(
            column_ids,
            row_ids,
            answers_list=(_to_coordinates(question, answer_coordinates))
        )

    def _find_tokens(self, text, segment):
        """Return start index of segment in text or None."""
        logging.info('text: %s %s', text, segment)
        for index in range(1 + len(text) - len(segment)):
            for seg_index, seg_token in enumerate(segment):
                if text[index + seg_index].piece != seg_token.piece:
                    break
            else:
                return index
        return None

    def _find_answer_coordinates_from_answer_text(
        self,
        tokenized_table,
        answer_text,
    ):
        """Returns all occurrences of answer_text in the table."""
        logging.info('answer text: %s', answer_text)
        for row_index, row in enumerate(tokenized_table.rows):
            if row_index == 0:
                # We don't search for answers in the header.
                continue
            for col_index, cell in enumerate(row):
                token_index = self._find_tokens(cell, answer_text)
                if token_index is not None:
                    yield TokenCoordinates(
                        row_index=row_index,
                        column_index=col_index,
                        token_index=token_index,
                    )

    def _find_answer_ids_from_answer_texts(
        self,
        column_ids,
        row_ids,
        tokenized_table,
        answer_texts,
    ):
        """Maps question with answer texts to the first matching token indexes."""
        answer_ids = [0] * len(column_ids)
        for answer_text in answer_texts:
            for coordinates in self._find_answer_coordinates_from_answer_text(
                tokenized_table,
                answer_text,
            ):
                # Maps answer coordinates to indexes this can fail if tokens / rows have
                # been pruned.
                indexes = list(
                    self._get_cell_token_indexes(
                        column_ids,
                        row_ids,
                        column_id=coordinates.column_index,
                        row_id=coordinates.row_index - 1,
                    ))
                indexes.sort()
                coordinate_answer_ids = []
                if indexes:
                    begin_index = coordinates.token_index + indexes[0]
                    end_index = begin_index + len(answer_text)
                    for index in indexes:
                        if index >= begin_index and index < end_index:
                            coordinate_answer_ids.append(index)
                if len(coordinate_answer_ids) == len(answer_text):
                    for index in coordinate_answer_ids:
                        answer_ids[index] = 1
                    break
        return answer_ids

    def _get_answer_ids(self, column_ids, row_ids, question, answer_coordinates):
        """Maps answer coordinates to token indexes."""
        answer_ids, missing_count = self._get_all_answer_ids(column_ids, row_ids,
                                                        [question],
                                                        answer_coordinates)

        if missing_count:
            raise ValueError("Couldn't find all answers")
        return answer_ids

    def get_answer_ids(self, column_ids, row_ids, tokenized_table, question, answer_texts_question, answer_coordinates_question):
        if self.update_answer_coordinates:
            return self._find_answer_ids_from_answer_texts(
                column_ids,
                row_ids,
                tokenized_table,
                answer_texts=[
                    self.tokenize(at)
                    for at in answer_texts_question
                ],
            )
        return self._get_answer_ids(column_ids, row_ids, question, answer_coordinates_question)

    #### End of everything related to label ids calculation ####

    def batch_encode_plus(self,
        table,
        queries: Union[
            List[TextInput],
            List[PreTokenizedInput],
            List[EncodedInput],
        ],
        answer_coordinates: Optional[List[Tuple]] = None,
        answer_texts: Optional[List[TextInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = True,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Tokenize and prepare for the model a list of one or more sequences related to a table.
        .. warning::
            This method is deprecated, ``__call__`` should be used instead.
        Args:
            queries (:obj:`List[str]`):
                Batch of sequences (queries) related to a table to be encoded.
                This is a list of string-sequences (see details in ``encode_plus``).
        """

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
        #     padding=padding,
        #     truncation=truncation,
        #     max_length=max_length,
        #     pad_to_multiple_of=pad_to_multiple_of,
        #     verbose=verbose,
        #     **kwargs,
        # )

        return self._batch_encode_plus(
            table=table,
            queries=queries,
            answer_coordinates=answer_coordinates,
            answer_texts=answer_texts,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

    def _batch_encode_plus(
        self,
        table,
        queries: Union[
            List[TextInput],
            List[PreTokenizedInput],
            List[EncodedInput],
        ],
        answer_coordinates: Optional[List[Tuple]] = None,
        answer_texts: Optional[List[TextInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = True,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:

        if return_offsets_mapping:
            raise NotImplementedError(
                "return_offset_mapping is not available when using Python tokenizers."
                "To use this feature, change your tokenizer to one deriving from "
                "transformers.PreTrainedTokenizerFast."
            )

        if "is_pretokenized" in kwargs:
            warnings.warn(
                "`is_pretokenized` is deprecated and will be removed in a future version, use `is_split_into_words` instead.",
                FutureWarning,
            )
        
        if "is_split_into_words" in kwargs:
            raise NotImplementedError("Currently TapasTokenizer only supports questions as strings.")

        batch_outputs = self._batch_prepare_for_model(
            table=table,
            queries=queries,
            answer_coordinates=answer_coordinates,
            answer_texts=answer_texts,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            return_tensors=return_tensors,
            verbose=verbose,
        )

        return BatchEncoding(batch_outputs)
    
    def _batch_prepare_for_model(
        self,
        table,
        queries: Union[
            List[TextInput],
            List[PreTokenizedInput],
            List[EncodedInput],
        ],
        answer_coordinates: Optional[List[Tuple]] = None,
        answer_texts: Optional[List[TextInput]] = None,
        add_special_tokens: bool = True,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = True,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ) -> BatchEncoding:
        """
        Prepares a sequence of strings (queries) related to a table so that it can be used by the model.
        It creates input ids, adds special tokens, truncates the table if overflowing (if the drop_rows_to_fit
        parameter is set to True) while taking into account the special tokens and manages a moving window 
        (with user defined stride) for overflowing tokens

        This function is based on prepare_for_model (but in Tapas, training examples depend on each other,
        so we defined it at a batch level)

        Args:
            table: Pandas dataframe
            queries: List of Strings, containing questions related to the table
        """

        if "return_lengths" in kwargs:
            if verbose:
                warnings.warn(
                    "The PreTrainedTokenizerBase.prepare_for_model `return_lengths` parameter is deprecated. "
                    "Please use `return_length` instead.",
                    FutureWarning,
                )
            return_length = kwargs["return_lengths"]

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        # padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
        #     padding=padding,
        #     truncation=truncation,
        #     max_length=max_length,
        #     pad_to_multiple_of=pad_to_multiple_of,
        #     verbose=verbose,
        #     **kwargs,
        # )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names
        
        encoded_inputs = {}

        if return_overflowing_tokens:
            raise ValueError("Overflowing tokens is currently not supported")

        if (answer_coordinates and not answer_texts) or (not answer_coordinates and answer_texts):
            raise ValueError("In case you provide answers, both answer_coordinates and answer_text should be provided") 

        add_loss_variables = None
        if answer_coordinates is not None and answer_texts is not None:
            assert len(answer_coordinates) == len(answer_texts)
            add_loss_variables = True
        
        # First, tokenize the table and get the number of rows and columns
        tokenized_table = self._tokenize_table(table)
        num_rows = self._get_num_rows(table, self.drop_rows_to_fit)
        num_columns = self._get_num_columns(table)
        
        # Second, create the input ids for every table + query pair (and all the other features). This is a list of lists
        features_examples = {}
        position_to_label_ids = {}
        for position, query in enumerate(queries):
            if isinstance(query, str):
                text_tokens = self.tokenize(query)
                # currently, padding is done within the _to_trimmed_features function 
                serialized_example, features = self._to_trimmed_features(
                                                                question=query,
                                                                table=table,
                                                                question_tokens=text_tokens,
                                                                tokenized_table=tokenized_table,
                                                                num_columns=num_columns,
                                                                num_rows=num_rows,
                                                                drop_rows_to_fit=self.drop_rows_to_fit)
                
                if add_loss_variables:
                    column_ids = serialized_example.column_ids
                    row_ids = serialized_example.row_ids

                    # create label ids from answer texts and coordinates
                    label_ids = self.get_answer_ids(column_ids, 
                                                    row_ids, 
                                                    tokenized_table, 
                                                    query, 
                                                    answer_texts[position],
                                                    answer_coordinates[position],
                                                    )
                    self._pad_to_seq_length(label_ids)
                    position_to_label_ids[position] = label_ids
                    features['label_ids'] = label_ids

                    if position == 0:
                        prev_label_ids = [0] * len(features["input_ids"])
                    else:
                        # TO DO: add prev label ids logic (see line 1118 in tf_example_utils.py)
                        prev_label_ids = position_to_label_ids[position - 1]
                    self._pad_to_seq_length(prev_label_ids)
                    features["prev_label_ids"] = prev_label_ids

                else:
                    prev_label_ids = [0] * len(features["input_ids"])
                    self._pad_to_seq_length(prev_label_ids)
                    features["prev_label_ids"] = prev_label_ids

                features_examples[position] = features
            else:
                raise ValueError(
                    "Query is not valid. Should be a string."
                )

        # Build output dictionnary
        encoded_inputs["input_ids"] = [features_examples[position]["input_ids"] for position in range(len(queries))]
        encoded_inputs["attention_mask"] = [features_examples[position]["attention_mask"] for position in range(len(queries))]
        
        token_types = ["segment_ids", "column_ids", "row_ids", "prev_label_ids", "column_ranks",
                            "inv_column_ranks", "numeric_relations"]
        token_type_ids = []
        for position in range(len(queries)):
            token_type_ids_example = []
            for token_idx in range(self.model_max_length):
                token_ids = []
                for type in token_types:
                    token_ids.append(features_examples[position][type][token_idx])
                token_type_ids_example.append(token_ids)
            # token_type_ids_example is a list of seq_length elements, each element being a list of 7 elements
            token_type_ids.append(token_type_ids_example)

        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = token_type_ids
        
        if add_loss_variables:
            encoded_inputs["label_ids"] = [features_examples[position]["label_ids"] for position in range(len(queries))]
            encoded_inputs["numeric_values"] = [features_examples[position]["numeric_values"] for position in range(len(queries))]
            encoded_inputs["numeric_values_scale"] = [features_examples[position]["numeric_values_scale"] for position in range(len(queries))]
            # to do: add aggregation function id, classification class index and answer (or should people prepare this themselves?)
        
        if return_special_tokens_mask:
            raise ValueError("Special tokens mask is currently not supported")

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])
        
        batch_outputs = BatchEncoding(encoded_inputs, tensor_type=return_tensors)

        return batch_outputs

    #### Everything related to converting logits to answers ####

    def _get_cell_token_probs(self, probabilities, segment_ids, row_ids, column_ids):
        for i, p in enumerate(probabilities):
            segment_id = segment_ids[i]
            col = column_ids[i] - 1
            row = row_ids[i] - 1
            if col >= 0 and row >= 0 and segment_id == 1:
                yield i, p

    def _get_mean_cell_probs(self, probabilities, segment_ids, row_ids, column_ids):
        """Computes average probability per cell, aggregating over tokens."""
        coords_to_probs = collections.defaultdict(list)  
        for i, prob in self._get_cell_token_probs(probabilities, segment_ids, row_ids, column_ids):
            col = column_ids[i] - 1
            row = row_ids[i] - 1
            coords_to_probs[(col, row)].append(prob)
        return {
            coords: torch.as_tensor(cell_probs).mean()
            for coords, cell_probs in coords_to_probs.items()
        }

    def _parse_coordinates(self, raw_coordinates):
        """Parses cell coordinates from text."""
        return [ast.literal_eval(x) for x in raw_coordinates]

    def convert_logits_to_answers(self, data, logits, logits_agg=None, logits_cls=None, cell_classification_threshold=0.5):
        # compute probabilities from token logits
        dist_per_token = torch.distributions.Bernoulli(logits=logits)
        probabilities = dist_per_token.probs * data["attention_mask"].type(torch.float32).to(dist_per_token.probs.device)
        
        token_types = ["segment_ids", "column_ids", "row_ids", "prev_label_ids", "column_ranks",
                                    "inv_column_ranks", "numeric_relations"] 
        
        # collect input_ids, segment ids, row ids and column ids of batch. Shape (batch_size, seq_len)
        input_ids = data["input_ids"]
        segment_ids = data["token_type_ids"][:,:,token_types.index("segment_ids")]
        row_ids = data["token_type_ids"][:,:,token_types.index("row_ids")]
        column_ids = data["token_type_ids"][:,:,token_types.index("column_ids")]

        # next, get answer coordinates for every example in the batch
        num_batch = input_ids.shape[0]
        answers_batch = []
        answer_coordinates_batch = []
        for i in range(num_batch):
            probabilities_example = probabilities[i].tolist()
            input_ids_example = input_ids[i].tolist()
            segment_ids_example = segment_ids[i]
            row_ids_example = row_ids[i]
            column_ids_example = column_ids[i]

            max_width = column_ids_example.max()
            max_height = row_ids_example.max()

            if (max_width == 0 and max_height == 0):
                continue
            
            cell_coords_to_prob = self._get_mean_cell_probs(probabilities_example, 
                                                    segment_ids_example.tolist(), 
                                                    row_ids_example.tolist(), 
                                                    column_ids_example.tolist())
        
            # Select the answers above the classification threshold.
            answer_coordinates = []
            for col in range(max_width):
                for row in range(max_height):
                    cell_prob = cell_coords_to_prob.get((col, row), None)
                    if cell_prob is not None:
                        if cell_prob > cell_classification_threshold:
                            answer_coordinates.append(str((row, col)))
            answer_coordinates = sorted(self._parse_coordinates(answer_coordinates))
            answer_coordinates_batch.append(answer_coordinates)

        output = answer_coordinates_batch
        
        if logits_agg is not None:
            aggregation_predictions = logits_agg.argmax(dim=-1)
            output = (output, aggregation_predictions.tolist())

        if logits_cls is not None:
            classification_predictions = logits_cls.argmax(dim=-1)
            output = output + (classification_predictions.tolist())
        
        return output

    #### End of everything related to converting logits to answers ####

            