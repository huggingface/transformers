from absl.testing import absltest
from absl.testing import parameterized
from interaction_pb2 import (Table, Question, Answer, Interaction)
from tapas_task_utils import (SupervisionMode, parse_all_question)
from google.protobuf import text_format


_Mode = SupervisionMode

def _set_float32_safe_interaction(interaction):
  new_interaction = Interaction()
  new_interaction.ParseFromString(interaction.SerializeToString())
  interaction.CopyFrom(new_interaction)


def _set_float32_safe_answer(answer):
  new_answer = Answer()
  new_answer.ParseFromString(answer.SerializeToString())
  answer.CopyFrom(new_answer)


class InteractionUtilsParserTest(parameterized.TestCase):

  def test_unambiguous_matching(self):
    interaction = text_format.Parse(
        """
      table {
        columns { text: "Column0" }
        rows { cells { text: "a" } }
        rows { cells { text: "ab" } }
        rows { cells { text: "b" } }
        rows { cells { text: "bc" } }
      }
      questions {
        answer {
          answer_texts: "a"
          answer_texts: "b"
        }
      }""", Interaction())

    question = parse_all_question(interaction.table,
                                                       interaction.questions[0],
                                                       _Mode.REMOVE_ALL)

    expected_answer = text_format.Parse(
        """
          answer_coordinates {
            row_index: 0
            column_index: 0
          }
          answer_coordinates {
            row_index: 2
            column_index: 0
          }
          answer_texts: "a"
          answer_texts: "b"
    """, Answer())

    self.assertEqual(expected_answer, question.answer)

  def test_ambiguous_matching(self):
    interaction = text_format.Parse(
        """
      table {
        columns { text: "Column0" }
        rows { cells { text: "a" } }
        rows { cells { text: "a" } }
      }
      questions {
        answer {
          answer_texts: "a"
          answer_texts: "a"
        }
      }""", Interaction())

    with self.assertRaises(ValueError):
      parse_all_question(interaction.table,
                                              interaction.questions[0],
                                              _Mode.REMOVE_ALL)

  def test_float_value(self):
    interaction = text_format.Parse(
        """
      table {
        columns { text: "Column0" }
        rows { cells { text: "a" } }
        rows { cells { text: "a" } }
      }
      questions {
        answer {
          answer_texts: "1.0"
        }
      }""", Interaction())

    question = parse_all_question(interaction.table,
                                                       interaction.questions[0],
                                                       _Mode.REMOVE_ALL)

    expected_answer = text_format.Parse(
        """
      answer_texts: "1.0"
      float_value: 1.0
    """, Answer())

    self.assertEqual(expected_answer, question.answer)

  @parameterized.named_parameters(
      ("no_filter", _Mode.NONE, """
        answer_coordinates {
          row_index: 0
          column_index: 0
        }
        answer_coordinates {
          row_index: 1
          column_index: 0
        }
        answer_texts: "2"
        aggregation_function: COUNT"""),
      ("remove_all", _Mode.REMOVE_ALL, """
        answer_texts: "2"
        float_value: 2.0"""),
  )
  def test_strategies(self, mode, expected_answer):
    interaction = text_format.Parse(
        """
      table {
        columns { text: "Column0" }
        rows { cells { text: "a" } }
        rows { cells { text: "b" } }
      }
      questions {
        answer {
          answer_coordinates {
            row_index: 0
            column_index: 0
          }
          answer_coordinates {
            row_index: 1
            column_index: 0
          }
          answer_texts: "2"
          aggregation_function: COUNT
        }
      }""", Interaction())

    question = parse_all_question(interaction.table,
                                                       interaction.questions[0],
                                                       mode)
    self.assertEqual(
        text_format.Parse(expected_answer, Answer()),
        question.answer)

  def test_set_answer_text_when_multiple_answers(self):
    interaction = text_format.Parse(
        """
      table {
        columns { text: "Column0" }
        rows { cells { text: "2008" } }
      }
      questions {
        answer {
          answer_texts: "1"
          answer_texts: "2"
          float_value: 2008.0
        }
      }""", Interaction())

    question = parse_all_question(interaction.table,
                                                       interaction.questions[0],
                                                       _Mode.REMOVE_ALL)

    expected_answer = text_format.Parse(
        """
          answer_coordinates {
            row_index: 0
            column_index: 0
          }
          answer_texts: "2008"
          float_value: 2008.0
    """, Answer())

    self.assertEqual(expected_answer, question.answer)

  def test_set_answer_text_strange_float_format_when_multiple_answers(self):
    interaction = text_format.Parse(
        """
      table {
        columns { text: "Column0" }
        rows { cells { text: "2008" } }
      }
      questions {
        answer {
          answer_texts: "1"
          answer_texts: "2"
          float_value: 2008.001
        }
      }""", Interaction())
    _set_float32_safe_interaction(interaction)
    question = parse_all_question(interaction.table,
                                                       interaction.questions[0],
                                                       _Mode.REMOVE_ALL)
    _set_float32_safe_interaction(interaction)
    expected_answer = text_format.Parse(
        """
          answer_texts: "2008.0009765625"
          float_value: 2008.001
    """, Answer())
    _set_float32_safe_answer(expected_answer)
    self.assertEqual(expected_answer, question.answer)

  def test_set_use_answer_text_when_single_float_answer(self):
    interaction = text_format.Parse(
        """
      table {
        columns { text: "Column0" }
        rows { cells { text: "2008.00000000000" } }
      }
      questions {
        answer {
          answer_texts: "2008.00000000000"
          float_value: 2008.0
        }
      }""", Interaction())

    question = parse_all_question(interaction.table,
                                                       interaction.questions[0],
                                                       _Mode.REMOVE_ALL)
    _set_float32_safe_interaction(interaction)
    expected_answer = text_format.Parse(
        """
          answer_coordinates {
            row_index: 0
            column_index: 0
          }
          answer_texts: "2008.00000000000"
          float_value: 2008.0
    """, Answer())
    _set_float32_safe_answer(expected_answer)
    self.assertEqual(expected_answer, question.answer)

  def test_set_answer_text_strict(self):
    interaction = text_format.Parse(
        """
      table {
        columns { text: "Year" }
        rows { cells { text: "2008" } }
        rows { cells { text: "2010" } }
        rows { cells { text: "2008" } }
      }
      questions {
        answer {
          answer_texts: "2008"
        }
      }""", Interaction())

    try:
      result = parse_all_question(
          interaction.table,
          interaction.questions[0],
          _Mode.REMOVE_ALL_STRICT,
      )
    except ValueError as error:
      result = str(error)
    self.assertEqual(
        result,
        "Cannot parse answer: "
        "[answer_coordinates: Found multiple cells for answers]",
    )


if __name__ == "__main__":
  absltest.main()