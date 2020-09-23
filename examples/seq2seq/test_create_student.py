import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.file_utils import cached_property
from transformers.testing_utils import require_torch


try:
    from .create_student import create_student, main
except ImportError:
    from create_student import create_student, main

TEACHER_MODEL = "sshleifer/bart-tiny-random"


@require_torch
class CreateStudentTester(unittest.TestCase):
    @cached_property
    def teacher_config(self):
        return AutoConfig.from_pretrained(TEACHER_MODEL)

    def test_same_encoder_small_decoder(self):
        student = create_student(TEACHER_MODEL, student_encoder_layers=None, student_decoder_layers=1)
        self.assertEqual(student.config.encoder_layers, self.teacher_config.encoder_layers)
        self.assertEqual(student.config.decoder_layers, 1)

    def test_same_decoder_small_encoder(self):
        student = create_student(TEACHER_MODEL, student_encoder_layers=1, student_decoder_layers=None)
        self.assertEqual(student.config.encoder_layers, 1)
        self.assertEqual(student.config.decoder_layers, self.teacher_config.encoder_layers)

    def test_small_enc_small_dec(self):
        student = create_student(TEACHER_MODEL, student_encoder_layers=1, student_decoder_layers=1)
        self.assertEqual(student.config.encoder_layers, 1)
        self.assertEqual(student.config.decoder_layers, 1)

    def test_raises_assert(self):
        with self.assertRaises(AssertionError):
            create_student(TEACHER_MODEL, student_encoder_layers=None, student_decoder_layers=None)

    def test_create_student_script(self):
        output_dir = tempfile.mkdtemp(prefix="student_model")
        testargs = f"""
            create_student.py
            --teacher_model_name_or_path {TEACHER_MODEL}
            --student_encoder_layers=1
            --student_decoder_layers=1
            --save_path {output_dir}
            """.split()

        with patch.object(sys, "argv", testargs):
            main()

        try:
            # check if model and tokenizer is saved
            model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)
            tok = AutoTokenizer.from_pretrained(output_dir)

            self.assertEqual(model.config.encoder_layers, 1)
            self.assertEqual(model.config.decoder_layers, 1)
        except Exception as e:
            self.fail(e)
        finally:
            shutil.rmtree(output_dir)
