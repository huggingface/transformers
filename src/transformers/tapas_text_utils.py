import re
import struct
from typing import Iterable, List, Text, Tuple, Union
import unicodedata
import six


def wtq_normalize(x):
  """Returns the normalized version of x.
  This normalization function is taken from WikiTableQuestions github, hence the
  wtq prefix. For more information, see
  https://github.com/ppasupat/WikiTableQuestions/blob/master/evaluator.py
  Args:
    x: the object (integer type or string) to normalize.
  Returns:
    A normalized string.
  """
  x = x if isinstance(x, six.text_type) else six.text_type(x)
  # Remove diacritics.
  x = "".join(
      c for c in unicodedata.normalize("NFKD", x)
      if unicodedata.category(c) != "Mn")
  # Normalize quotes and dashes.
  x = re.sub(u"[‘’´`]", "'", x)
  x = re.sub(u"[“”]", '"', x)
  x = re.sub(u"[‐‑‒–—−]", "-", x)
  x = re.sub(u"[‐]", "", x)
  while True:
    old_x = x
    # Remove citations.
    x = re.sub(u"((?<!^)\\[[^\\]]*\\]|\\[\\d+\\]|[•♦†‡*#+])*$", "",
               x.strip())
    # Remove details in parenthesis.
    x = re.sub(u"(?<!^)( \\([^)]*\\))*$", "", x.strip())
    # Remove outermost quotation mark.
    x = re.sub(u'^"([^"]*)"$', r"\1", x.strip())
    if x == old_x:
      break
  # Remove final '.'.
  if x and x[-1] == ".":
    x = x[:-1]
  # Collapse whitespaces and convert to lower case.
  x = re.sub(r"\s+", " ", x, flags=re.U).lower().strip()
  x = re.sub("<[^<]+?>", "", x)
  x = x.replace("\n", " ")
  return x


_TOKENIZER = re.compile(r"\w+|[^\w\s]+", re.UNICODE)


def tokenize_string(x):
  return list(_TOKENIZER.findall(x.lower()))


# List of string normalization functions to be applied in order. We go from
# simplest to more complex normalization procedures.
STRING_NORMALIZATIONS = (
    lambda x: x,
    lambda x: x.lower(),
    tokenize_string,
    wtq_normalize,
)


def _split_thousands(delimiter, value):
  split = value.split(delimiter)
  return len(split) > 1 and any(map(lambda x: len(x) == 3, split))


def convert_to_float(value):
  """Converts value to a float using a series of increasingly complex heuristics.
  Args:
    value: object that needs to be converted. Allowed types include
      float/int/strings.
  Returns:
    A float interpretation of value.
  Raises:
    ValueError if the float conversion of value fails.
  """
  if isinstance(value, float):
    return value
  if isinstance(value, int):
    return float(value)
  if not isinstance(value, six.string_types):
    raise ValueError("Argument value is not a string. Can't parse it as float")
  sanitized = value

  try:
    # Example: 1,000.7
    if "." in sanitized and "," in sanitized:
      return float(sanitized.replace(",", ""))
    # 1,000
    if "," in sanitized and _split_thousands(",", sanitized):
      return float(sanitized.replace(",", ""))
    # 5,5556
    if "," in sanitized and sanitized.count(",") == 1 and not _split_thousands(
        ",", sanitized):
      return float(sanitized.replace(",", "."))
    # 0.0.0.1
    if sanitized.count(".") > 1:
      return float(sanitized.replace(".", ""))
    # 0,0,0,1
    if sanitized.count(",") > 1:
      return float(sanitized.replace(",", ""))
    return float(sanitized)
  except ValueError:
    # Avoid adding the sanitized value in the error message.
    raise ValueError("Unable to convert value to float")


def to_float32(v):
  """If v is a float reduce precision to that of a 32 bit float."""
  if not isinstance(v, float):
    return v
  return struct.unpack("!f", struct.pack("!f", v))[0]


def get_sequence_id(example_id, annotator):
  if "-" in annotator:
    raise ValueError('"-" not allowed in annotator.')
  return f"{example_id}-{annotator}"


def get_question_id(sequence_id, position):
  return f"{sequence_id}_{position}"