from tqdm import tqdm
import collections
import logging
import os
import json
import numpy as np

from ...tokenization_bert import BasicTokenizer, whitespace_tokenize
from .utils import DataProcessor, InputExample, InputFeatures
from ...file_utils import is_tf_available

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)

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


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
        # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def squad_convert_examples_to_features(examples, tokenizer, max_seq_length,
                                       doc_stride, max_query_length, is_training,
                                       cls_token_at_end=True,
                                       cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                       sequence_a_segment_id=0, sequence_b_segment_id=1,
                                       cls_token_segment_id=0, pad_token_segment_id=0,
                                       mask_padding_with_zero=True,
                                       sequence_a_is_doc=False):
    """Loads a data file into a list of `InputBatch`s."""

    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token

    # Defining helper methods    
    unique_id = 1000000000

    features = []
    new_features = []
    for (example_index, example) in enumerate(tqdm(examples)):

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in example.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        if is_training:
            # Get start and end position
            answer_length = len(example.answer_text)
            start_position = char_to_word_offset[example.start_position]
            end_position = char_to_word_offset[example.start_position + answer_length - 1]

            # If the answer cannot be found in the text, then skip this example.
            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
            if actual_text.find(cleaned_answer_text) == -1:
                logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                continue

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        spans = []
        
        truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, max_length=max_query_length)
        sequence_added_tokens = tokenizer.max_len - tokenizer.max_len_single_sentence 
        sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair 

        encoded_dict = tokenizer.encode_plus(
            truncated_query if not sequence_a_is_doc else all_doc_tokens, 
            all_doc_tokens if not sequence_a_is_doc else truncated_query, 
            max_length=max_seq_length, 
            padding_strategy='right',
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
            return_overflowing_tokens=True, 
            truncation_strategy='only_second' if not sequence_a_is_doc else 'only_first'
        )

        ids = encoded_dict['input_ids']
        non_padded_ids = ids[:ids.index(tokenizer.pad_token_id)] if tokenizer.pad_token_id in ids else ids
        paragraph_len = min(len(all_doc_tokens) - len(spans) * doc_stride, max_seq_length - len(truncated_query) - sequence_pair_added_tokens)
        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(truncated_query) + sequence_added_tokens + i if not sequence_a_is_doc else i 
            token_to_orig_map[index] = tok_to_orig_index[0 + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = 0
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)
        # print("YESSIR", len(spans) * doc_stride < len(all_doc_tokens), "overflowing_tokens" in encoded_dict)

        while len(spans) * doc_stride < len(all_doc_tokens) and "overflowing_tokens" in encoded_dict:
            overflowing_tokens = encoded_dict["overflowing_tokens"]
            encoded_dict = tokenizer.encode_plus(
                truncated_query if not sequence_a_is_doc else overflowing_tokens, 
                overflowing_tokens if not sequence_a_is_doc else truncated_query, 
                max_length=max_seq_length, 
                return_overflowing_tokens=True, 
                padding_strategy='right',
                stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
                truncation_strategy='only_second' if not sequence_a_is_doc else 'only_first'
            )
            ids = encoded_dict['input_ids']
            # print("Ids computes; position of the first padding", ids.index(tokenizer.pad_token_id) if tokenizer.pad_token_id in ids else None)

            # print(encoded_dict["input_ids"].index(tokenizer.pad_token_id) if tokenizer.pad_token_id in encoded_dict["input_ids"] else None)
            # print(len(spans) * doc_stride, len(all_doc_tokens))
            

            # Length of the document without the query
            paragraph_len = min(len(all_doc_tokens) - len(spans) * doc_stride, max_seq_length - len(truncated_query) - sequence_pair_added_tokens)

            if tokenizer.pad_token_id in encoded_dict['input_ids']: 
                non_padded_ids = encoded_dict['input_ids'][:encoded_dict['input_ids'].index(tokenizer.pad_token_id)]
            else:
                non_padded_ids = encoded_dict['input_ids']

            tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

            token_to_orig_map = {}
            for i in range(paragraph_len):
                index = len(truncated_query) + sequence_added_tokens + i if not sequence_a_is_doc else i 
                token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

            encoded_dict["paragraph_len"] = paragraph_len
            encoded_dict["tokens"] = tokens
            encoded_dict["token_to_orig_map"] = token_to_orig_map
            encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
            encoded_dict["token_is_max_context"] = {}
            encoded_dict["start"] = len(spans) * doc_stride
            encoded_dict["length"] = paragraph_len

            spans.append(encoded_dict)

        for doc_span_index in range(len(spans)):
            for j in range(spans[doc_span_index]["paragraph_len"]):
                is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
                index = j if sequence_a_is_doc else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
                spans[doc_span_index]["token_is_max_context"][index] = is_max_context

        for span in spans:
            # Identify the position of the CLS token
            cls_index = span['input_ids'].index(tokenizer.cls_token_id)

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = np.array(span['token_type_ids'])

            p_mask = np.minimum(p_mask, 1)

            if not sequence_a_is_doc:
                # Limit positive values to one
                p_mask = 1 - p_mask

            p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

            # Set the CLS index to '0'
            p_mask[cls_index] = 0

            new_features.append(NewSquadFeatures(
                span['input_ids'],
                span['attention_mask'],
                span['token_type_ids'],
                cls_index,
                p_mask.tolist(),

                example_index=example_index,
                unique_id=unique_id,
                paragraph_len=span['paragraph_len'],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"]
            ))

            unique_id += 1

        # tokenize ...
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(doc_tokens) - 1:
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
            # print("Start offset is", start_offset, len(all_doc_tokens), "length is", length)
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

            span_is_impossible = example.is_impossible if hasattr(example, "is_impossible") else False
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

            # if example_index < 20:
            #     logger.info("*** Example ***")
            #     logger.info("unique_id: %s" % (unique_id))
            #     logger.info("example_index: %s" % (example_index))
            #     logger.info("doc_span_index: %s" % (doc_span_index))
            #     logger.info("tokens: %s" % str(tokens))
            #     logger.info("token_to_orig_map: %s" % " ".join([
            #         "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
            #     logger.info("token_is_max_context: %s" % " ".join([
            #         "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
            #     ]))
            #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #     logger.info(
            #         "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            #     logger.info(
            #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            #     if is_training and span_is_impossible:
            #         logger.info("impossible example")
            #     if is_training and not span_is_impossible:
            #         answer_text = " ".join(tokens[start_position:(end_position + 1)])
            #         logger.info("start_position: %d" % (start_position))
            #         logger.info("end_position: %d" % (end_position))
            #         logger.info(
            #             "answer: %s" % (answer_text))

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

        assert len(features) == len(new_features)

    assert len(features) == len(new_features)
    for i in range(len(features)):
        feature, new_feature = features[i], new_features[i]
        
        input_ids = [f if f not in [3,4,5] else 0 for f in feature.input_ids ]
        input_mask = feature.input_mask
        segment_ids = feature.segment_ids
        cls_index = feature.cls_index
        p_mask = feature.p_mask
        example_index = feature.example_index
        paragraph_len = feature.paragraph_len
        token_is_max_context = feature.token_is_max_context
        tokens = feature.tokens
        token_to_orig_map = feature.token_to_orig_map
              
        new_input_ids = [f if f not in [3,4,5] else 0 for f in new_feature.input_ids]
        new_input_mask = new_feature.attention_mask
        new_segment_ids = new_feature.token_type_ids
        new_cls_index = new_feature.cls_index
        new_p_mask = new_feature.p_mask
        new_example_index = new_feature.example_index
        new_paragraph_len = new_feature.paragraph_len
        new_token_is_max_context = new_feature.token_is_max_context
        new_tokens = new_feature.tokens
        new_token_to_orig_map = new_feature.token_to_orig_map

        assert input_ids == new_input_ids
        assert input_mask == new_input_mask
        assert segment_ids == new_segment_ids
        assert cls_index == new_cls_index
        assert p_mask == new_p_mask
        assert example_index == new_example_index
        assert paragraph_len == new_paragraph_len
        assert token_is_max_context == new_token_is_max_context

        tokens = [t if tokenizer.convert_tokens_to_ids(t) is not tokenizer.unk_token_id else tokenizer.unk_token for t in tokens]

        assert tokens == new_tokens
        assert token_to_orig_map == new_token_to_orig_map


    return new_features


def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples


class SquadV1Processor(DataProcessor):
    """Processor for the SQuAD data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return NewSquadExample(
            tensor_dict['id'].numpy(),
            tensor_dict['question'].numpy().decode('utf-8'),
            tensor_dict['context'].numpy().decode('utf-8'),
            tensor_dict['answers']['text'].numpy().decode('utf-8'),
            tensor_dict['answers']['answers_start'].numpy().decode('utf-8'),
            tensor_dict['title'].numpy().decode('utf-8')
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "train-v1.1.json"), "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "dev-v1.1.json"), "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, input_data, set_type):
        """Creates examples for the training and dev sets."""
        
        is_training = set_type == "train"
        examples = []
        for entry in input_data:
            title = entry['title']
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    answer_text = None
                    if is_training:
                        if (len(qa["answers"]) != 1):
                            raise ValueError(
                                "For training, each question should have exactly 1 answer.")
                        answer = qa["answers"][0]
                        answer_text = answer['text']
                        start_position = answer['answer_start']

                    example = NewSquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position=start_position,
                        title=title
                    )
                    examples.append(example)
        return examples
        


class NewSquadExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 context_text,
                 answer_text,
                 start_position,
                 title):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.start_position = start_position
        self.title = title


class NewSquadFeatures(object):
    """
    Single squad example features to be fed to a model.
    Those features are model-specific.
    """

    def __init__(self,
                 input_ids,
                 attention_mask,
                 token_type_ids,
                 cls_index,
                 p_mask,
                 
                 example_index,
                 unique_id,
                 paragraph_len,
                 token_is_max_context,
                 tokens,
                 token_to_orig_map
        ):
        self.input_ids = input_ids 
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


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
        print(self.example_index == other.example_index)
        print(self.input_ids == other.input_ids)
        print(self.input_mask == other.attention_mask)
        print(self.p_mask == other.p_mask)
        print(self.paragraph_len == other.paragraph_len)
        print(self.segment_ids == other.token_type_ids)
        print(self.token_is_max_context == other.token_is_max_context)
        print(self.token_to_orig_map == other.token_to_orig_map)
        print(self.tokens == other.tokens)

        return self.example_index == other.example_index and \
                self.input_ids == other.input_ids and \
                self.input_mask == other.attention_mask and \
                self.p_mask == other.p_mask and \
                self.paragraph_len == other.paragraph_len and \
                self.segment_ids == other.token_type_ids and \
                self.token_is_max_context == other.token_is_max_context and \
                self.token_to_orig_map == other.token_to_orig_map and \
                self.tokens == other.tokens