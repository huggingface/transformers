import tempfile
import unittest

from make_student import create_student_by_copying_alternating_layers
from transformers import AutoConfig
from transformers.file_utils import cached_property
from transformers.testing_utils import require_torch


TINY_BART = "sshleifer/bart-tiny-random"
TINY_T5 = "patrickvonplaten/t5-tiny-random"


@require_torch
class MakeStudentTester(unittest.TestCase):
    @cached_property
    def teacher_config(self):
        return AutoConfig.from_pretrained(TINY_BART)

    def test_valid_t5(self):
        student, *_ = create_student_by_copying_alternating_layers(TINY_T5, tempfile.mkdtemp(), e=1, d=1)
        self.assertEqual(student.config.num_hidden_layers, 1)

    def test_asymmetric_t5(self):
        student, *_ = create_student_by_copying_alternating_layers(TINY_T5, tempfile.mkdtemp(), e=1, d=None)

    def test_same_decoder_small_encoder(self):
        student, *_ = create_student_by_copying_alternating_layers(TINY_BART, tempfile.mkdtemp(), e=1, d=None)
        self.assertEqual(student.config.encoder_layers, 1)
        self.assertEqual(student.config.decoder_layers, self.teacher_config.encoder_layers)

    def test_small_enc_small_dec(self):
        student, *_ = create_student_by_copying_alternating_layers(TINY_BART, tempfile.mkdtemp(), e=1, d=1)
        self.assertEqual(student.config.encoder_layers, 1)
        self.assertEqual(student.config.decoder_layers, 1)

    def test_raises_assert(self):
        with self.assertRaises(AssertionError):
            create_student_by_copying_alternating_layers(TINY_BART, tempfile.mkdtemp(), e=None, d=None)
