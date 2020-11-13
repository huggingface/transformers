import os
from absl import logging
import numpy as np
import collections
import six
import frozendict
import csv
import ast
import enum
from scipy import (optimize)

from tapas_text_utils import (STRING_NORMALIZATIONS, convert_to_float, to_float32, get_sequence_id, get_question_id)
from interaction_pb2 import (Table, Question, Answer, Interaction)
from tapas_file_utils import (list_directory, make_directories)
from tapas_wtq_utils import (convert)
####################################################################################
# CONSTANTS
####################################################################################

_AggregationFunction = Answer.AggregationFunction

_ID = 'id'
_ANNOTATOR = 'annotator'
_POSITION = 'position'
_QUESTION = 'question'
_TABLE_FILE = 'table_file'
_ANSWER_TEXT = 'answer_text'
_ANSWER_COORDINATES = 'answer_coordinates'
_AGGREGATION = 'aggregation'
_ANSWER_FLOAT_VALUE = 'float_answer'
_ANSWER_CLASS_INDEX = 'class_index'

####################################################################################
# HELPERS AND MODES
####################################################################################


class SupervisionMode(enum.Enum):
    # Don't filter out any supervised information.
    NONE = 0
    # Remove all the supervised signals and recompute them by parsing answer
    # texts.
    REMOVE_ALL = 2

    REMOVE_ALL_STRICT = 3


_CLEAR_FIELDS = frozendict.frozendict({
    SupervisionMode.REMOVE_ALL: [
        "answer_coordinates", "float_value", "aggregation_function"
    ],
    SupervisionMode.REMOVE_ALL_STRICT: [
        "answer_coordinates", "float_value", "aggregation_function"
    ]
})


def get_supervision_modes(task):
    """Gets the correct supervision mode for each task."""
    if task == "WIKISQL":
        return {
            'train.tsv': SupervisionMode.REMOVE_ALL,
            'dev.tsv': SupervisionMode.NONE,
            'test.tsv': SupervisionMode.NONE
        }
    if task == "WTQ":
        return collections.defaultdict(lambda: SupervisionMode.REMOVE_ALL)
    if task in [
        "SQA",
        "WIKISQL_SUPERVISED",
        "TABFACT",
    ]:
        return collections.defaultdict(lambda: SupervisionMode.NONE)
    raise ValueError(f'Unknown task: {task.name}')


def get_interaction_dir(output_dir):
    return os.path.join(output_dir, 'interactions')

####################################################################################
# CREATING DATA N INTERACTION
####################################################################################


def create_interactions(task, input_dir,
                        output_dir):
    """Converts original task data to interactions.
    Interactions will be written to f'{output_dir}/interactions'. Other files
    might be written as well.
    Args:
      task: The current task.
      input_dir: Data with original task data.
      output_dir: Outputs are written to this directory.
    """
    if task == "SQA":
        tsv_dir = input_dir
    elif task == "WTQ":
        convert(input_dir, output_dir)
        tsv_dir = output_dir
        return
    else:
        raise ValueError(f'Unknown task: {task.name}')
    create_all_interactions(
        get_supervision_modes(task),
        tsv_dir,
        get_interaction_dir(output_dir),
    )


def _read_interactions(input_dir):
    """Reads interactions from TSV files."""
    filenames = [
        fn for fn in list_directory(input_dir) if fn.endswith('.tsv')
    ]
    interaction_dict = {}
    for filename in filenames:
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r") as file_handle:
            try:
                interactions = read_from_tsv_file(file_handle)
                interaction_dict[filename] = interactions
            except KeyError as ke:
                logging.error("Can't read interactions from file: %s (%s)", filepath, ke)
    return interaction_dict


def create_all_interactions(supervision_modes,
                            input_dir,
                            output_dir):
    """Converts data in SQA format to Interaction protos.
    Args:
      supervision_modes: Import for WikiSQL, decide if supervision is removed.
      input_dir: SQA data.
      output_dir: Where interactions will be written.
    """
    make_directories(output_dir)

    interaction_dict = _read_interactions(input_dir)
    _add_tables(input_dir, interaction_dict)
    _parse_questions(interaction_dict, supervision_modes,
                     os.path.join(output_dir, 'report.tsv'))


def _add_tables(input_dir,
                interaction_dict):
    """Adds table protos to all interactions."""
    table_files = set()
    for interactions in interaction_dict.values():
        for interaction in interactions:
            table_files.add(interaction.table.table_id)

    table_dict = {}
    for index, table_file in enumerate(sorted(table_files)):
        logging.log_every_n(logging.INFO, 'Read %4d / %4d table files', 100, index,
                            len(table_files))
        table_path = os.path.join(input_dir, table_file)
        with open(table_path, "r") as table_handle:
            table = Table()
            rows = list(csv.reader(table_handle))
            headers, rows = rows[0], rows[1:]

            for header in headers:
                table.columns.add().text = header

            for row in rows:
                new_row = table.rows.add()
                for cell in row:
                    new_row.cells.add().text = cell

            table.table_id = table_file
            table_dict[table_file] = table

    for interactions in interaction_dict.values():
        for interaction in interactions:
            interaction.table.CopyFrom(table_dict[interaction.table.table_id])


def _parse_questions(interaction_dict,
                     supervision_modes,
                     report_filename):
    """Adds numeric value spans to all questions."""
    counters = collections.defaultdict(collections.Counter)
    for key, interactions in interaction_dict.items():
        for interaction in interactions:
            questions = []
            for original_question in interaction.questions:
                try:
                    question = parse_all_question(
                        interaction.table, original_question, supervision_modes[key])
                    counters[key]['valid'] += 1
                except ValueError as exc:
                    question = Question()
                    question.CopyFrom(original_question)
                    question.answer.is_valid = False
                    counters[key]['failed'] += 1
                    counters[key]['failed-' + str(exc)] += 1

                questions.append(question)

            del interaction.questions[:]
            interaction.questions.extend(questions)


def _has_single_float_answer_equal_to(question, target):
    """Returns true if the question has a single answer whose value equals to target."""
    if len(question.answer.answer_texts) != 1:
        return False
    try:
        float_value = convert_to_float(question.answer.answer_texts[0])
        # In general answer_float is derived by applying the same conver_to_float
        # function at interaction creation time, hence here we use exact match to
        # avoid any false positive.
        return to_float32(float_value) == to_float32(target)
    except ValueError:
        return False


def parse_all_question(table,
                       question, mode):
    """Parses answer_text field of question to populate additional fields needed to create TF examples.
    Args:
      table: a Table message, needed to compute the answer coordinates.
      question: a Question message, that will be modified (even on unsuccesful
        parsing).
      mode: See SupervisionMode enum for more information.
    Returns:
      A Question message with answer_coordinates or float_value field populated.
    Raises:
      ValueError if we cannot parse correctly the question message.
    """

    if mode == SupervisionMode.NONE:
        return question

    clear_fields = _CLEAR_FIELDS.get(mode, None)
    if clear_fields is None:
        raise ValueError(f"Mode {mode.name} is not supported")

    return _parse_question(
        table,
        question,
        clear_fields,
        discard_ambiguous_examples=mode == SupervisionMode.REMOVE_ALL_STRICT,
    )


def _parse_question(
    table,
    original_question,
    clear_fields,
    discard_ambiguous_examples,
):
    """Parses question's answer_texts fields to possibly populate additional fields.
    Args:
      table: a Table message, needed to compute the answer coordinates.
      original_question: a Question message containing answer_texts.
      clear_fields: A list of strings indicating which fields need to be cleared
        and possibly repopulated.
      discard_ambiguous_examples: If true, discard ambiguous examples.
    Returns:
      A Question message with answer_coordinates or float_value field populated.
    Raises:
      ValueError if we cannot parse correctly the question message.
    """

    question = Question()
    question.CopyFrom(original_question)

    # If we have a float value signal we just copy its string representation to
    # the answer text (if multiple answers texts are present OR the answer text
    # cannot be parsed to float OR the float value is different), after clearing
    # this field.
    if "float_value" in clear_fields and question.answer.HasField("float_value"):
        if not _has_single_float_answer_equal_to(question,
                                                 question.answer.float_value):
            del question.answer.answer_texts[:]
            float_value = float(question.answer.float_value)
            if float_value.is_integer():
                number_str = str(int(float_value))
            else:
                number_str = str(float_value)
            question.answer.answer_texts.append(number_str)

    if not question.answer.answer_texts:
        raise ValueError("No answer_texts provided")

    for field_name in clear_fields:
        question.answer.ClearField(field_name)

    error_message = ""
    if not question.answer.answer_coordinates:
        try:
            _parse_answer_coordinates(
                table,
                question.answer,
                discard_ambiguous_examples,
            )
        except ValueError as exc:
            error_message += "[answer_coordinates: {}]".format(str(exc))
            if discard_ambiguous_examples:
                raise ValueError(f"Cannot parse answer: {error_message}")

    if not question.answer.HasField("float_value"):
        try:
            _parse_answer_float(question.answer)
        except ValueError as exc:
            error_message += "[float_value: {}]".format(str(exc))

    # Raises an exception if we cannot set any of the two fields.
    if not question.answer.answer_coordinates and not question.answer.HasField(
            "float_value"):
        raise ValueError("Cannot parse answer: {}".format(error_message))

    return question


def _parse_answer_coordinates(table,
                              answer,
                              discard_ambiguous_examples):
    """Populates answer_coordinates using answer_texts.
    Args:
      table: a Table message, needed to compute the answer coordinates.
      answer: an Answer message that will be modified on success.
      discard_ambiguous_examples: If true discard if answer has multiple matches.
    Raises:
      ValueError if the conversion fails.
    """
    del answer.answer_coordinates[:]
    cost_matrix = _compute_cost_matrix(
        table,
        answer,
        discard_ambiguous_examples,
    )
    if cost_matrix is None:
        return
    row_indices, column_indices = optimize.linear_sum_assignment(
        cost_matrix)
    for _ in row_indices:
        answer.answer_coordinates.add()
    for row_index in row_indices:
        flatten_position = column_indices[row_index]
        row_coordinate = flatten_position // len(table.columns)
        column_coordinate = flatten_position % len(table.columns)
        answer.answer_coordinates[row_index].row_index = row_coordinate
        answer.answer_coordinates[row_index].column_index = column_coordinate


def _compute_cost_matrix(
    table,
    answer,
    discard_ambiguous_examples,
):
    """Computes cost matrix."""
    for index, normalize_fn in enumerate(STRING_NORMALIZATIONS):
        try:
            result = _compute_cost_matrix_inner(
                table,
                answer,
                normalize_fn,
                discard_ambiguous_examples,
            )
            if result is None:
                continue
            return result
        except ValueError:
            if index == len(STRING_NORMALIZATIONS) - 1:
                raise
    return None


def _compute_cost_matrix_inner(
    table,
    answer,
    normalize,
    discard_ambiguous_examples,
):
    """Returns a cost matrix M where the value M[i,j] contains a matching cost from answer i to cell j.
    The matrix is a binary matrix and -1 is used to indicate a possible match from
    a given answer_texts to a specific cell table. The cost matrix can then be
    usedto compute the optimal assignments that minimizes the cost using the
    hungarian algorithm (see scipy.optimize.linear_sum_assignment).
    Args:
      table: a Table message.
      answer: an Answer message.
      normalize: a function that normalizes a string.
      discard_ambiguous_examples: If true discard if answer has multiple matches.
    Raises:
      ValueError if:
        - we cannot correctly construct the cost matrix or the text-cell
        assignment is ambiguous.
        - we cannot find a matching cell for a given answer_text.
    Returns:
      A numpy matrix with shape (num_answer_texts, num_rows * num_columns).
    """
    max_candidates = 0
    num_cells = len(table.rows) * len(table.columns)
    num_candidates = np.zeros((len(table.rows), len(table.columns)))
    cost_matrix = np.zeros((len(answer.answer_texts), num_cells))

    for index, answer_text in enumerate(answer.answer_texts):
        found = 0
        for row, column in _find_matching_coordinates(table, answer_text,
                                                      normalize):
            found += 1
            cost_matrix[index, (row * len(table.columns)) + column] = -1
            num_candidates[row, column] += 1
            max_candidates = max(max_candidates, num_candidates[row, column])
        if found == 0:
            return None
        if discard_ambiguous_examples and found > 1:
            raise ValueError("Found multiple cells for answers")

    # TODO(piccinno): Shall we allow ambiguous assignments?
    if max_candidates > 1:
        raise ValueError("Assignment is ambiguous")

    return cost_matrix


def _find_matching_coordinates(table, answer_text,
                               normalize):
    normalized_text = normalize(answer_text)
    for row_index, row in enumerate(table.rows):
        for column_index, cell in enumerate(row.cells):
            if normalized_text == normalize(cell.text):
                yield (row_index, column_index)


def _parse_answer_float(answer):
    if len(answer.answer_texts) > 1:
        raise ValueError("Cannot convert to multiple answers to single float")
    float_value = convert_to_float(answer.answer_texts[0])
    answer.float_value = float_value


def _parse_answer_text(answer_text, answer):
    """Populates the answer_texts field of `answer` by parsing `answer_text`.
    Args:
      answer_text: A string representation of a Python list of strings.
        For example: "[u'test', u'hello', ...]"
      answer: an Answer object.
    """
    try:
        for value in ast.literal_eval(answer_text):
            answer.answer_texts.append(value)
    except SyntaxError:
        raise ValueError('Unable to evaluate %s' % answer_text)


def read_from_tsv_file(
        file_handle):
    """Parses a TSV file in SQA format into a list of interactions.
    Args:
      file_handle:  File handle of a TSV file in SQA format.
    Returns:
      Questions grouped into interactions.
    """
    questions = {}
    for row in csv.DictReader(file_handle, delimiter='\t'):
        sequence_id = get_sequence_id(row[_ID], row[_ANNOTATOR])
        key = sequence_id, row[_TABLE_FILE]
        if key not in questions:
            questions[key] = {}

        position = int(row[_POSITION])

        answer = Answer()
        _parse_answer_coordinates(row[_ANSWER_COORDINATES], answer)
        _parse_answer_text(row[_ANSWER_TEXT], answer)

        if _AGGREGATION in row:
            agg_func = row[_AGGREGATION].upper().strip()
            if agg_func:
                answer.aggregation_function = _AggregationFunction.Value(agg_func)
        if _ANSWER_FLOAT_VALUE in row:
            float_value = row[_ANSWER_FLOAT_VALUE]
            if float_value:
                answer.float_value = float(float_value)
        if _ANSWER_CLASS_INDEX in row:
            class_index = row[_ANSWER_CLASS_INDEX]
            if class_index:
                answer.class_index = int(class_index)

        questions[key][position] = Question(
            id=get_question_id(sequence_id, position),
            original_text=row[_QUESTION],
            answer=answer)

    interactions = []
    for (sequence_id, table_file), question_dict in sorted(
            questions.items(), key=lambda sid: sid[0]):
        question_list = [
            question for _, question in sorted(
                question_dict.items(), key=lambda pos: pos[0])
        ]
        interactions.append(
            Interaction(
                id=sequence_id,
                questions=question_list,
                table=Table(table_id=table_file)))
    return interactions
