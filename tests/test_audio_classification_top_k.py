import unittest
import numpy as np
import pytest
from transformers import pipeline, AutoConfig

from transformers.testing_utils import require_torch