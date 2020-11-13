import enum
from dataclasses import dataclass
from typing import Text, Union, Tuple, Iterable, List, Any, Dict, Callable, Optional, Set
import re
import itertools
import collections
import datetime
import math


class Relation(enum.Enum):
  HEADER_TO_CELL = 1  # Connects header to cell.
  CELL_TO_HEADER = 2  # Connects cell to header.
  QUERY_TO_HEADER = 3  # Connects query to headers.
  QUERY_TO_CELL = 4  # Connects query to cells.
  ROW_TO_CELL = 5  # Connects row to cells.
  CELL_TO_ROW = 6  # Connects cells to row.
  EQ = 7  # Annotation value is same as cell value
  LT = 8  # Annotation value is less than cell value
  GT = 9  # Annotation value is greater than cell value

@dataclass
class Date():
  year: Optional[int] = None
  month: Optional[int] = None
  day: Optional[int] = None

@dataclass
class NumericValue:
  float_value: Optional[float] = None
  date: Optional[Date] = None

@dataclass
class NumericValueSpan:
  begin_index: int = None
  end_index: int = None
  values: List[NumericValue] = None
  
@dataclass
class Cell:
  text: Text 
  numeric_value: Optional[NumericValue] = None

# Constants for parsing date expressions.
# Masks that specify (by a bool) which of (year, month, day) will be populated.
_DateMask = collections.namedtuple('_DateMask', ['year', 'month', 'day'])

_YEAR = _DateMask(True, False, False)
_YEAR_MONTH = _DateMask(True, True, False)
_YEAR_MONTH_DAY = _DateMask(True, True, True)
_MONTH = _DateMask(False, True, False)
_MONTH_DAY = _DateMask(False, True, True)

# Pairs of patterns to pass to 'datetime.strptime' and masks specifying which
# fields will be set by the corresponding pattern.
_DATE_PATTERNS = (('%B', _MONTH), ('%Y', _YEAR), ('%Ys', _YEAR),
                  ('%b %Y', _YEAR_MONTH), ('%B %Y', _YEAR_MONTH),
                  ('%B %d', _MONTH_DAY), ('%b %d', _MONTH_DAY), ('%d %b',
                                                                 _MONTH_DAY),
                  ('%d %B', _MONTH_DAY), ('%B %d, %Y', _YEAR_MONTH_DAY),
                  ('%d %B %Y', _YEAR_MONTH_DAY), ('%m-%d-%Y', _YEAR_MONTH_DAY),
                  ('%Y-%m-%d', _YEAR_MONTH_DAY), ('%Y-%m', _YEAR_MONTH),
                  ('%B %Y', _YEAR_MONTH), ('%d %b %Y', _YEAR_MONTH_DAY),
                  ('%Y-%m-%d', _YEAR_MONTH_DAY), ('%b %d, %Y', _YEAR_MONTH_DAY),
                  ('%d.%m.%Y', _YEAR_MONTH_DAY),
                  ('%A, %b %d', _MONTH_DAY), ('%A, %B %d', _MONTH_DAY))

# This mapping is used to convert date patterns to regex patterns.
_FIELD_TO_REGEX = (
    ('%A', r'\w+'),  # Weekday as locale’s full name.
    ('%B', r'\w+'),  # Month as locale’s full name.
    ('%Y', r'\d{4}'),  #  Year with century as a decimal number.
    ('%b', r'\w{3}'),  # Month as locale’s abbreviated name.
    ('%d', r'\d{1,2}'),  # Day of the month as a zero-padded decimal number.
    ('%m', r'\d{1,2}'),  # Month as a zero-padded decimal number.
)


def _process_date_pattern(dp):
  """Compute a regex for each date pattern to use as a prefilter."""
  pattern, mask = dp
  regex = pattern
  regex = regex.replace('.', re.escape('.'))
  regex = regex.replace('-', re.escape('-'))
  regex = regex.replace(' ', r'\s+')
  for field, field_regex in _FIELD_TO_REGEX:
    regex = regex.replace(field, field_regex)
  # Make sure we didn't miss any of the fields.
  assert '%' not in regex, regex
  return pattern, mask, re.compile('^' + regex + '$')


def _process_date_patterns():
  return tuple(_process_date_pattern(dp) for dp in _DATE_PATTERNS)


_PROCESSED_DATE_PATTERNS = _process_date_patterns()

_MAX_DATE_NGRAM_SIZE = 5

# Following DynSp:
# https://github.com/Microsoft/DynSP/blob/master/util.py#L414.
_NUMBER_WORDS = [
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
    'nine', 'ten', 'eleven', 'twelve'
]

_ORDINAL_WORDS = [
    'zeroth', 'first', 'second', 'third', 'fourth', 'fith', 'sixth', 'seventh',
    'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth'
]

_ORDINAL_SUFFIXES = ['st', 'nd', 'rd', 'th']

_NUMBER_PATTERN = re.compile(r'((^|\s)[+-])?((\.\d+)|(\d+(,\d\d\d)*(\.\d*)?))')

# Following DynSp:
# https://github.com/Microsoft/DynSP/blob/master/util.py#L293.
_MIN_YEAR = 1700
_MAX_YEAR = 2016

_INF = float('INF')


def _get_numeric_value_from_date(
    date, mask):
  """Converts date (datetime Python object) to a NumericValue object with a Date object value."""
  if date.year < _MIN_YEAR or date.year > _MAX_YEAR:
    raise ValueError('Invalid year: %d' % date.year)

  new_date = Date()
  if mask.year:
    new_date.year = date.year
  if mask.month:
    new_date.month = date.month
  if mask.day:
    new_date.day = date.day
  return NumericValue(date=new_date)


def _get_span_length_key(span):
  """Sorts span by decreasing length first and incresing first index second."""
  return span[1] - span[0], -span[0]


def _get_numeric_value_from_float(value):
  """Converts float (Python) to a NumericValue object with a float value."""
  return NumericValue(float_value=value)


# Doesn't parse ordinal expressions such as '18th of february 1655'.
def _parse_date(text):
  """Attempts to format a text as a standard date string (yyyy-mm-dd)."""
  text = re.sub(r'Sept\b', 'Sep', text)
  for in_pattern, mask, regex in _PROCESSED_DATE_PATTERNS:
    if not regex.match(text):
      continue
    try:
      date = datetime.datetime.strptime(text, in_pattern).date()
    except ValueError:
      continue
    try:
      return _get_numeric_value_from_date(date, mask)
    except ValueError:
      continue
  return None


def _parse_number(text):
  """Parses simple cardinal and ordinals numbers."""
  for suffix in _ORDINAL_SUFFIXES:
    if text.endswith(suffix):
      text = text[:-len(suffix)]
      break
  text = text.replace(',', '')
  try:
    value = float(text)
  except ValueError:
    return None
  if math.isnan(value):
    return None
  if value == _INF:
    return None
  return value


def normalize_for_match(text):
  return " ".join(text.lower().split())


def get_all_spans(text,
                  max_ngram_length):
  """Split a text into all possible ngrams up to 'max_ngram_length'.
  Split points are white space and punctuation.
  Args:
    text: Text to split.
    max_ngram_length: maximal ngram length.
  Yields:
    Spans, tuples of begin-end index.
  """
  start_indexes = []
  for index, char in enumerate(text):
    if not char.isalnum():
      continue
    if index == 0 or not text[index - 1].isalnum():
      start_indexes.append(index)
    if index + 1 == len(text) or not text[index + 1].isalnum():
      for start_index in start_indexes[-max_ngram_length:]:
        yield start_index, index + 1


def parse_text(text):
  """Extracts longest number and date spans.
  Args:
    text: text to annotate.
  Returns:
    List of longest numeric value spans.
  """
  span_dict = collections.defaultdict(list)
  for match in _NUMBER_PATTERN.finditer(text):
    span_text = text[match.start():match.end()]
    number = _parse_number(span_text)
    if number is not None:
      span_dict[match.span()].append(_get_numeric_value_from_float(number))

  for begin_index, end_index in get_all_spans(
      text, max_ngram_length=1):
    if (begin_index, end_index) in span_dict:
      continue
    span_text = text[begin_index:end_index]

    number = _parse_number(span_text)
    if number is not None:
      span_dict[begin_index, end_index].append(
          _get_numeric_value_from_float(number))
    for number, word in enumerate(_NUMBER_WORDS):
      if span_text == word:
        span_dict[begin_index, end_index].append(
            _get_numeric_value_from_float(float(number)))
        break
    for number, word in enumerate(_ORDINAL_WORDS):
      if span_text == word:
        span_dict[begin_index, end_index].append(
            _get_numeric_value_from_float(float(number)))
        break

  for begin_index, end_index in get_all_spans(
      text, max_ngram_length=_MAX_DATE_NGRAM_SIZE):
    span_text = text[begin_index:end_index]
    date = _parse_date(span_text)
    if date is not None:
      span_dict[begin_index, end_index].append(date)

  spans = sorted(
      span_dict.items(),
      key=lambda span_value: _get_span_length_key(span_value[0]),
      reverse=True)
  selected_spans = []
  for span, value in spans:
    for selected_span, _ in selected_spans:
      if selected_span[0] <= span[0] and span[1] <= selected_span[1]:
        break
    else:
      selected_spans.append((span, value))

  selected_spans.sort(key=lambda span_value: span_value[0][0])

  numeric_value_spans = []
  for span, values in selected_spans:
    numeric_value_spans.append(
        NumericValueSpan(
            begin_index=span[0], end_index=span[1], values=values))
  return numeric_value_spans


_PrimitiveNumericValue = Union[
    float, Tuple[Optional[float], Optional[float], Optional[float]]]
_SortKeyFn = Callable[[NumericValue], Tuple[float, Ellipsis]]

_DATE_TUPLE_SIZE = 3

NUMBER_TYPE = 'number'
DATE_TYPE = 'date'


def _get_value_type(numeric_value):
  if numeric_value.float_value is not None:
    return NUMBER_TYPE
  elif numeric_value.date is not None:
    return DATE_TYPE
  raise ValueError('Unknown type: %s' % numeric_value)


def _get_value_as_primitive_value(
    numeric_value):
  """Maps a NumericValue proto to a float or tuple of float."""
  if numeric_value.float_value is not None:
    return numeric_value.float_value
  if numeric_value.date is not None:
    date = numeric_value.date
    value_tuple = [None, None, None]
    # All dates fields are cased to float to produce a simple primitive value.
    if date.year is not None:
      value_tuple[0] = float(date.year)
    if date.month is not None:
      value_tuple[1] = float(date.month)
    if date.day is not None:
      value_tuple[2] = float(date.day)
    return tuple(value_tuple)
  raise ValueError('Unknown type: %s' % numeric_value)


def _get_all_types(
    numeric_values):
  return {_get_value_type(value) for value in numeric_values}


def get_numeric_sort_key_fn(
    numeric_values):
  """Creates a function that can be used as a sort key or to compare the values.
  Maps to primitive types and finds the biggest common subset.
  Consider the values "05/05/2010" and "August 2007".
  With the corresponding primitive values (2010.,5.,5.) and (2007.,8., None).
  These values can be compared by year and date so we map to the sequence
  (2010., 5.), (2007., 8.).
  If we added a third value "2006" with primitive value (2006., None, None),
  we could only compare by the year so we would map to (2010.,), (2007.,)
  and (2006.,).
  Args:
   numeric_values: Values to compare.
  Returns:
   A function that can be used as a sort key function (mapping numeric values
   to a comparable tuple).
  Raises:
    ValueError if values don't have a common type or are not comparable.
  """
  value_types = _get_all_types(numeric_values)
  if len(value_types) != 1:
    raise ValueError('No common value type in %s' % numeric_values)

  value_type = next(iter(value_types))
  if value_type == NUMBER_TYPE:
    # Primitive values are simple floats, nothing to do here.
    return _get_value_as_primitive_value

  # The type can only be Date at this point which means the primitive type
  # is a float triple.
  valid_indexes = set(range(_DATE_TUPLE_SIZE))

  for numeric_value in numeric_values:
    value = _get_value_as_primitive_value(numeric_value)
    assert isinstance(value, tuple)
    for tuple_index, inner_value in enumerate(value):
      if inner_value is None:
        valid_indexes.discard(tuple_index)

  if not valid_indexes:
    raise ValueError('No common value in %s' % numeric_values)

  def _sort_key_fn(numeric_value):
    value = _get_value_as_primitive_value(numeric_value)
    return tuple(value[index] for index in valid_indexes)

  return _sort_key_fn


def _get_numeric_values(text):
  """Parses text and returns numeric values."""
  numeric_spans = parse_text(text)
  return itertools.chain(*(span.values for span in numeric_spans))


def _parse_column_values(
    table,
    col_index):
  """Parses text in column and returns a dict mapping row_index to values.
  Args: 
  table: Pandas dataframe
  col_index: integer, indicating the index of the column to get the numeric values of 
  """
  index_to_values = {}
  for row_index, row in table.iterrows():
    text = normalize_for_match(row[col_index])
    index_to_values[row_index] = list(_get_numeric_values(text))
  return index_to_values


def add_numeric_values_to_question(question):
  """Adds numeric value spans to a question."""
  question_original_text = question
  question = normalize_for_match(question)
  numeric_spans = parse_text(question) 

  return (question, numeric_spans)


def get_numeric_relation(value, other_value, sort_key_fn):
    """Compares two values and returns their relation or None."""
    value = sort_key_fn(value)
    other_value = sort_key_fn(other_value)
    if value == other_value:
      return Relation.EQ
    if value < other_value:
      return Relation.LT
    if value > other_value:
      return Relation.GT
    return None


