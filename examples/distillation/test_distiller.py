import argparse
import unittest

from .scripts.extract import extract


test_bart_args = argparse.Namespace(
    model_type="bart",
    # model_name='bart-large-cnn', '',
    model_name="sshleifer/bart-tiny-random",
    vocab_transform=True,
    dump_checkpoint="test_dir/bart_tiny.pth",
)
test_roberta_args = argparse.Namespace(
    model_type="roberta",
    # model_name='bart-large-cnn', '',
    model_name="roberta-base",
    vocab_transform=True,
    dump_checkpoint="test_dir/roberta_tiny.pth",
)


class TestDistiller(unittest.TestCase):
    def test_bart_extraction(self):
        extract(test_bart_args)

    def test_roberta_extraction(self):
        extract(test_roberta_args)
