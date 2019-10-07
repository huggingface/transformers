# coding=utf-8

from __future__ import absolute_import, division, print_function
import gzip
import json
import logging
import math
import collections
import glob
import enum
import random
import argparse
import pickle
import time

from pytorch_transformers.tokenization_bert import whitespace_tokenize

logger = logging.getLogger(__name__)

TextSpan = collections.namedtuple("TextSpan", "token_positions text")


class AnswerType(enum.IntEnum):
    """Type of NQ answer."""
    UNKNOWN = 0
    YES = 1
    NO = 2
    SHORT = 3
    LONG = 4


import numpy as np


def dev_answer(json_example, is_training):
    if is_training:
        if len(json_example['annotations']) != 1:
            raise ValueError(
                'Train set json_examples should have a single annotation.')
        annotation = json_example['annotations'][0]
        has_long_answer = annotation['long_answer']['start_byte'] >= 0
        has_short_answer = annotation[
                               'short_answers'] or annotation['yes_no_answer'] != 'NONE'
    else:
        if len(json_example['annotations']) != 5:
            raise ValueError('Dev set json_examples should have five annotations.')
        has_long_answer = sum([
            annotation['long_answer']['start_byte'] >= 0
            for annotation in json_example['annotations']
        ]) >= 2
        has_short_answer = sum([
            bool(annotation['short_answers']) or
            annotation['yes_no_answer'] != 'NONE'
            for annotation in json_example['annotations']
        ]) >= 2

    long_answers = [
        a['long_answer']
        for a in json_example['annotations']
        if a['long_answer']['start_byte'] >= 0 and has_long_answer
    ]
    short_answers = [
        (a['short_answers'], a["long_answer"]["candidate_index"])
        for a in json_example['annotations']
        if a['short_answers'] and has_short_answer
    ]
    return has_long_answer, has_short_answer, long_answers, short_answers


class Answer(collections.namedtuple("Answer", ["type", "text", "offset"])):
    """Answer record.
    An Answer contains the type of the answer and possibly the text (for
    long) as well as the offset (for extractive).
    """

    def __new__(cls, type_, text=None, offset=None):
        return super(Answer, cls).__new__(cls, type_, text, offset)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class NqExample(object):
    """A single training/test example."""

    def __init__(self,
                 example_id,  # int example_id
                 qas_id,  # string example_id
                 questions,  # question tokens
                 doc_tokens,  # doc_tokens list, including real words and tags
                 doc_tokens_map=None,  # same size with doc_tokens, and -1 is the tag, >1 is the origin token id
                 answer=None,  # ["type","text","offset"]
                 start_position=None,  # the start/end origin token id (according to the doc_tokens_map)
                 end_position=None
                 ):
        self.example_id = example_id
        self.qas_id = qas_id
        self.questions = questions
        self.doc_tokens = doc_tokens
        self.doc_tokens_map = doc_tokens_map
        self.answer = answer
        self.start_position = start_position
        self.end_position = end_position


def make_nq_answer(contexts, answer):
    """Makes an Answer object following NQ conventions.
    Args:
    contexts: string containing the context
    answer: dictionary with `span_start` and `input_text` fields
    Returns:
    an Answer object. If the Answer type is YES or NO or LONG, the text
    of the answer is the long answer. If the answer type is UNKNOWN, the text of
    the answer is empty.
    """
    start = answer["span_start"]
    end = answer["span_end"]
    input_text = answer["input_text"]
    # lq added
    long_id = answer["candidate_id"]
    if (answer["candidate_id"] == -1 or start >= len(contexts) or end > len(contexts)):
        answer_type = AnswerType.UNKNOWN
        start = 0
        end = 1
        long_id = -1  # lq added
    elif input_text.lower() == "yes":
        answer_type = AnswerType.YES
    elif input_text.lower() == "no":
        answer_type = AnswerType.NO
    elif input_text.lower() == "long":
        answer_type = AnswerType.LONG
    else:
        answer_type = AnswerType.SHORT
    return Answer(answer_type, text=contexts[start:end], offset=start)


def get_candidate_text(e, idx):
    """Returns a text representation of the candidate at the given index."""
    # No candidate at this index.
    if idx < 0 or idx >= len(e["long_answer_candidates"]):
        return TextSpan([], "")
    # This returns an actual candidate.
    return get_text_span(e, e["long_answer_candidates"][idx])


def should_skip_context(e, idx):
    if (args.skip_nested_contexts and
            not e["long_answer_candidates"][idx]["top_level"]):
        return True
    elif not get_candidate_text(e, idx).text.strip():
        # Skip empty contexts.
        return True
    else:
        return False


def get_candidate_type(e, idx):
    """Returns the candidate's type: Table, Paragraph, List or Other."""
    c = e["long_answer_candidates"][idx]
    first_token = e["document_tokens"][c["start_token"]]["token"]
    if first_token == "<Table>":
        return "Table"
    elif first_token == "<P>":
        return "Paragraph"
    elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
        return "List"
    elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
        return "Other"
    else:
        logging.warning("Unknoww candidate type found: %s", first_token)
        return "Other"


def candidates_iter(e):
    """Yield's the candidates that should not be skipped in an example."""
    for idx, c in enumerate(e["long_answer_candidates"]):
        if should_skip_context(e, idx):
            continue
        yield idx, c


def add_candidate_types_and_positions(e):
    """Adds type and position info to each candidate in the document."""
    counts = collections.defaultdict(int)
    for idx, c in candidates_iter(e):
        context_type = get_candidate_type(e, idx)
        if counts[context_type] < args.max_position:
            counts[context_type] += 1
        c["type_and_position"] = "[%s=%d]" % (context_type, counts[context_type])


def has_long_answer(a):
    return (a["long_answer"]["start_token"] >= 0 and
            a["long_answer"]["end_token"] >= 0)


def token_to_char_offset(e, candidate_idx, token_idx):
    """Converts a token index to the char offset within the candidate."""
    c = e["long_answer_candidates"][candidate_idx]
    char_offset = 0
    for i in range(c["start_token"], token_idx):
        t = e["document_tokens"][i]
        if not t["html_token"]:
            token = t["token"].replace(" ", "")
            char_offset += len(token) + 1
    return char_offset


def get_first_annotation(e):
    """Returns the first short or long answer in the example.
    Args:
    e: (dict) annotated example.
    Returns:
    annotation: (dict) selected annotation
    annotated_idx: (int) index of the first annotated candidate.
    annotated_sa: (tuple) char offset of the start and end token
        of the short answer. The end token is exclusive.
    """
    positive_annotations = sorted(
        [a for a in e["annotations"] if has_long_answer(a)],
        key=lambda a: a["long_answer"]["candidate_index"])

    for a in positive_annotations:
        if a["short_answers"]:
            idx = a["long_answer"]["candidate_index"]
            start_token = a["short_answers"][0]["start_token"]
            end_token = a["short_answers"][-1]["end_token"]
            return a, idx, (token_to_char_offset(e, idx, start_token),
                            token_to_char_offset(e, idx, end_token) - 1)

    for a in positive_annotations:
        idx = a["long_answer"]["candidate_index"]
        return a, idx, (-1, -1)

    return None, -1, (-1, -1)


def get_text_span(example, span):
    """Returns the text in the example's document in the given token span."""
    token_positions = []
    tokens = []
    for i in range(span["start_token"], span["end_token"]):
        t = example["document_tokens"][i]
        if not t["html_token"]:
            token_positions.append(i)
            token = t["token"].replace(" ", "")
            tokens.append(token)
    return TextSpan(token_positions, " ".join(tokens))


def get_candidate_type_and_position(e, idx):
    """Returns type and position info for the candidate at the given index."""
    if idx == -1:
        return "[NoLongAnswer]"
    else:
        return e["long_answer_candidates"][idx]["type_and_position"]


def create_example_from_jsonl(line):
    """Creates an NQ example from a given line of JSON."""
    e = json.loads(line, object_pairs_hook=collections.OrderedDict)
    add_candidate_types_and_positions(e)

    context_idxs = [-1]
    context_list = [{"id": -1, "type": get_candidate_type_and_position(e, -1)}]
    context_list[-1]["text_map"], context_list[-1]["text"] = (get_candidate_text(e, -1))
    for idx, _ in candidates_iter(e):
        context = {"id": idx, "type": get_candidate_type_and_position(e, idx)}
        context["text_map"], context["text"] = get_candidate_text(e, idx)
        context_idxs.append(idx)
        context_list.append(context)
        # if len(context_list) >= args.max_contexts:
        #     break

    has_long, has_short, longs, shorts = dev_answer(e, False)
    candidate_answers = []
    if (not has_long) and (not has_short):
        is_impossibile = True
    else:
        is_impossibile = False
        if has_short:
            for (a, idx) in shorts:
                annotation, annotated_idx, annotated_sa = (a, idx, (token_to_char_offset(e, idx, a[0]["start_token"]),
                                                                    token_to_char_offset(e, idx,
                                                                                         a[0]["end_token"]) - 1))
                answer = {
                    "candidate_id": idx,
                    # "long_answer_text":get_candidate_text(e, annotated_idx).text,
                    "span_text": get_candidate_text(e, annotated_idx).text[annotated_sa[0]:annotated_sa[1]],
                    "span_start": annotated_sa[0],
                    "span_end": annotated_sa[1],
                    "input_text": "short"}
                if e["long_answer_candidates"][idx]["top_level"] == True:
                    candidate_answers.append(answer)

        else:
            for a in longs:
                idx = a["candidate_index"]
                annotation, annotated_idx, annotated_sa = (a, idx, (token_to_char_offset(e, idx, a["start_token"]),
                                                                    token_to_char_offset(e, idx, a["end_token"]) - 1))
                answer = {
                    "candidate_id": a["candidate_index"],
                    "span_text": get_candidate_text(e, annotated_idx).text,
                    "span_start": 0,
                    "span_end": len(get_candidate_text(e, annotated_idx).text),
                    "input_text": "long"}
                if e["long_answer_candidates"][idx]["top_level"] == True:
                    candidate_answers.append(answer)

    question = {"input_text": e["question_text"]}
    # Assemble example.
    example = {
        "name": e["document_title"],
        "id": str(e["example_id"]),
        "questions": [question],
        "answers": [],
        # "has_correct_context": annotated_idx in context_idxs
    }
    single_map = []
    single_context = []
    offset = 0
    for context in context_list:
        single_map.extend([-1, -1])
        single_context.append("[ContextId=%d] %s" %
                              (context["id"], context["type"]))
        offset += len(single_context[-1]) + 1
        for answer in candidate_answers:
            if context["id"] == answer["candidate_id"]:
                answer["span_start"] += offset
                answer["span_end"] += offset
        if context["text"]:
            single_map.extend(context["text_map"])
            single_context.append(context["text"])
            offset += len(single_context[-1]) + 1

    example["contexts"] = " ".join(single_context)
    example["contexts_map"] = single_map
    example["answers"] = candidate_answers
    example["answer_state_lq"] = (has_long, has_short)
    example["long_answer_candidates"]=e["long_answer_candidates"]
    return example


def read_nq_entry(entry, is_training):
    """Converts a NQ entry into a list of NqExamples."""

    def is_whitespace(c):
        return c in " \t\r\n" or ord(c) == 0x202F

    examples = []
    contexts_id = entry["id"]
    contexts = entry["contexts"]
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in contexts:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    questions = []
    for i, question in enumerate(entry["questions"]):
        qas_id = "{}".format(contexts_id)
        question_text = question["input_text"]
        start_position = None
        end_position = None
        answer = None
        if is_training:
            answer_dict = entry["answers"][i]
            answer = make_nq_answer(contexts, answer_dict)

            # For now, only handle extractive, yes, and no.
            if answer is None or answer.offset is None:
                continue
            start_position = char_to_word_offset[answer.offset]
            end_position = char_to_word_offset[answer.offset + len(answer.text) - 1]

            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(whitespace_tokenize(answer.text))
            if actual_text.find(cleaned_answer_text) == -1:
                logger.warning("Could not find answer: '%s' vs. '%s'", actual_text,
                               cleaned_answer_text)
                continue

        questions.append(question_text)
        example = NqExample(
            example_id=int(contexts_id),
            qas_id=qas_id,
            questions=questions[:],
            doc_tokens=doc_tokens,
            doc_tokens_map=entry.get("contexts_map", None),
            answer=answer,
            start_position=start_position,
            end_position=end_position)
        examples.append(example)
    return examples


def read_nq_examples_from_jsonl(input_file):
    """
    revised of read_nq_examples (tf run_nq.py) by lq
    :param input_file: a single jsonl.gz
    :return: NQExamples
    """
    input_data = []

    def _open(path):
        if path.endswith(".gz"):
            return gzip.open(path, "r")
        else:
            print("wrong file")
            exit()

    # logger.info("Reading: %s", input_file)
    with _open(input_file) as input_jsonl:
        for line in input_jsonl:
            input_data.append(create_example_from_jsonl(line))  # char-level
    return input_data  # examples


def nq_2_squad_dev(nqexample_list):
    skips = []
    squadexample_list = []
    for e in nqexample_list:
        if len(e['questions']) != 1:
            skips.append(e)
            continue
        has_long, has_short = e["answer_state_lq"]

        if ((not has_short) and (not has_long)) or (len(e["answers"]) == 0):
            is_impossible = True
            squand_answer = []
        else:
            is_impossible = False
            squand_answer = [
                {"text": ans["span_text"],
                 "answer_start": ans["span_start"],
                 "nq_candidate_id": ans["candidate_id"],
                 "nq_span_end": ans["span_end"],
                 "nq_input_text": ans["input_text"],
                 "nq_span_start": ans["span_start"],
                 "nq_span_text": ans["span_text"]}
                for ans in e["answers"]]
        long_candidates = []
        for c in e["long_answer_candidates"]:
            long_candidates.append((c["start_token"],c["end_token"]))


        squad_example = {
            "title": e['name'],
            "paragraphs": [{
                'context': e["contexts"],
                'nq_context_map': e["contexts_map"],
                'nq_long_candidates':long_candidates,
                'qas': [{
                    'question': e["questions"][0]["input_text"],
                    'id': e['id'],
                    'is_impossible': is_impossible,
                    'answers': squand_answer}
                ]
            }]
        }
        if not is_impossible:
            for ans in squad_example["paragraphs"][0]["qas"][0]["answers"]:
                start = ans["answer_start"]
                end = start + len(ans["text"])
                if ans["text"] != squad_example["paragraphs"][0]["context"][start:end]:
                    # print("wrong",len(squad_example["paragraphs"][0]["qas"][0]["answers"]))
                    # print("wrong",len(ans))
                    print("-------")
                    print(ans["nq_candidate_id"])
                    print(ans["text"])
                    print(squad_example["paragraphs"][0]["context"][start:end])
                # else:
                # print("right",len(squad_example["paragraphs"][0]["qas"][0]["answers"]))
        squadexample_list.append(squad_example)
    return squadexample_list, skips

def pickle2json(pickle_path,json_path):
    with open(pickle_path,'rb') as r:
        data = pickle.load(r)
        # print(type(data))
        with open(json_path,'wb') as w:
            datas = json.dumps({'data':data,'version':'nq'})
            # print(type(datas))
            w.write(datas.encode())
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--skip_nested_contexts", default=True, type=bool,
                        help="Completely ignore context that are not top level nodes in the page.")
    parser.add_argument("--max_position", default=50, type=int,
                        help="Maximum context position for which to generate special tokens.")
    parser.add_argument("--max_contexts", default=48, type=int,
                        help="Maximum number of contexts to output for an example.")

    parser.add_argument("--nq_dev_dir", default=None, type=str, required=True,
                        help="the director path of dev nq_gzip.")
    parser.add_argument("--saved_name", default=None, type=str, required=True,
                        help="the output filename of the squad_format json file for dev set.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="the output dir to save the nq_format.pk, squad_format.pk, and squad_format.json file.")
    args = parser.parse_args()
    # -----------------------------------------------------------------------------------------------------
    input_file_dir = args.nq_dev_dir
    output_json_path = args.output_dir + "/" + args.saved_name + "_squad2format.json"
    output_squad_pk_path = args.output_dir + "/" + args.saved_name + "_squad2format.pk"
    output_nq_pk_path = args.output_dir + "/" + args.saved_name + "_nqformat.pk"
    # -----------------------------------------------------------------------------------------------------
    input_paths = []
    for path in glob.glob("{}/*.gz".format(input_file_dir)):
        input_paths.append(path)
    print("Input dir:{}".format(input_file_dir))
    print("Containg gzip files: {}".format("\t\n".join(input_paths)))

    # ---------------------------process----------------------
    total_nq_examples = []
    from tqdm import tqdm
    for file in tqdm(input_paths):
        nq_examples = read_nq_examples_from_jsonl(file)
        total_nq_examples.extend(nq_examples)
    pickle.dump(total_nq_examples, open(output_nq_pk_path, "wb"))
    print("Finish: gzip to nq_format.pk, and saved to {}".format(output_nq_pk_path))
    squadexample_list, skips = nq_2_squad_dev(total_nq_examples)
    pickle.dump(squadexample_list, open(output_squad_pk_path, "wb"))
    print("Finish: nq_format.pk to squad2_format.pk, and saved to {}".format(output_squad_pk_path))

    pickle2json(output_squad_pk_path, output_json_path)
    print("Finish: squad2_format.pk to squad2format.json, and saved to {}".format(output_json_path))
    # -------------------------stastics---------
    count_a = 0
    count_b = 0
    count_c = 0
    for e in total_nq_examples:
        has_long, has_short = e["answer_state_lq"]
        if (not has_long) and (not has_short):
            count_a += 1
        else:
            if len(e["answers"]) == 0:
                count_b += 1
            else:
                count_c += 1
    print(count_a + count_b, count_c)
    # ----------------------has or no answer--------------
    count_no_answer = 0
    count_has_answer = 0
    for e in squadexample_list:
        if e["paragraphs"][0]["qas"][0]["is_impossible"]:  # no answer
            count_no_answer += 1
            if len(e["paragraphs"][0]["qas"][0]["answers"]) != 0:
                print("!")
        else:
            count_has_answer += 1
            if len(e["paragraphs"][0]["qas"][0]["answers"]) == 0:
                print("errors")
    #--------------------------------------------------------
    print("Has Ans example: {}".format(count_has_answer))
    print("No Ans example: {}".format(count_no_answer))
    print("Finished")
