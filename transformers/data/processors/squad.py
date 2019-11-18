from tqdm import tqdm
import collections
import logging
import os

from .utils import DataProcessor, InputExample, InputFeatures
from ...file_utils import is_tf_available

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)

def squad_convert_examples_to_features(examples, tokenizer, max_seq_length,
                                       doc_stride, max_query_length, is_training,
                                       cls_token_at_end=False,
                                       cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                       sequence_a_segment_id=0, sequence_b_segment_id=1,
                                       cls_token_segment_id=0, pad_token_segment_id=0,
                                       mask_padding_with_zero=True,
                                       sequence_a_is_doc=False):
    """Loads a data file into a list of `InputBatch`s."""

    # Defining helper methods
    def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)
    def _check_is_max_context(doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index
    
    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(tqdm(examples)):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0

            # XLNet: P SEP Q SEP CLS
            # Others: CLS Q SEP P SEP
            if not sequence_a_is_doc:
                # Query
                tokens += query_tokens
                segment_ids += [sequence_a_segment_id] * len(query_tokens)
                p_mask += [1] * len(query_tokens)

                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                if not sequence_a_is_doc:
                    segment_ids.append(sequence_b_segment_id)
                else:
                    segment_ids.append(sequence_a_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length

            if sequence_a_is_doc:
                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

                tokens += query_tokens
                segment_ids += [sequence_b_segment_id] * len(query_tokens)
                p_mask += [1] * len(query_tokens)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    if sequence_a_is_doc:
                        doc_offset = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index

            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and span_is_impossible:
                    logger.info("impossible example")
                if is_training and not span_is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))

            features.append(
                SquadFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible))
            unique_id += 1

    return features

class SquadFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __eq__(self, other):
        return self.cls_index == other.cls_index and \
                self.doc_span_index == other.doc_span_index and \
                self.end_position == other.end_position and \
                self.example_index == other.example_index and \
                self.input_ids == other.input_ids and \
                self.input_mask == other.input_mask and \
                self.is_impossible == other.is_impossible and \
                self.p_mask == other.p_mask and \
                self.paragraph_len == other.paragraph_len and \
                self.segment_ids == other.segment_ids and \
                self.start_position == other.start_position and \
                self.token_is_max_context == other.token_is_max_context and \
                self.token_to_orig_map == other.token_to_orig_map and \
                self.tokens == other.tokens and \
                self.unique_id == other.unique_id