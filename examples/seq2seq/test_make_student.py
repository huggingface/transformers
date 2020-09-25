import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from make_student import create_student_by_copying_alternating_layers
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.file_utils import cached_property
from transformers.testing_utils import require_torch


TEACHER_MODEL = "sshleifer/bart-tiny-random"


@require_torch
class CreateStudentTester(unittest.TestCase):
    @cached_property
    def teacher_config(self):
        return AutoConfig.from_pretrained(TEACHER_MODEL)

    def test_same_encoder_small_decoder(self):
        student, *_ = create_student_by_copying_alternating_layers(TEACHER_MODEL, e=None, d=1)
        self.assertEqual(student.config.encoder_layers, self.teacher_config.encoder_layers)
        self.assertEqual(student.config.decoder_layers, 1)

    def test_same_decoder_small_encoder(self):
        student, *_ = create_student_by_copying_alternating_layers("patrickvonplaten/t5-tiny-random", e=1, d=None)
        self.assertEqual(student.config.encoder_layers, 1)
        self.assertEqual(student.config.decoder_layers, self.teacher_config.encoder_layers)

    def test_small_enc_small_dec(self):
        student, *_ = create_student_by_copying_alternating_layers(TEACHER_MODEL, e=1, d=1)
        self.assertEqual(student.config.encoder_layers, 1)
        self.assertEqual(student.config.decoder_layers, 1)

    def test_raises_assert(self):
        with self.assertRaises(AssertionError):
            create_student_by_copying_alternating_layers(TEACHER_MODEL, e=None, d=None)
