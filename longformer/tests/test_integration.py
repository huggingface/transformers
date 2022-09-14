import torch
import unittest
from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size
from transformers import RobertaTokenizer


class TestEndToEnd(unittest.TestCase):

    def _run_test(self, device, dtype, attention_mode):

        config = LongformerConfig.from_pretrained(
            '/net/s3/s2-research/beltagy/longformer/model_release/longformer-base-4096/config.json')
        config.attention_mode = attention_mode
        model = Longformer.from_pretrained(
            '/net/s3/s2-research/beltagy/longformer/model_release/longformer-base-4096/pytorch_model.bin',
            config=config)
        model = model.eval()

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        tokenizer.model_max_length = 4096

        SAMPLE_TEXT = ' '.join(['Hello world! '] * 1025)  # long input document
        token_ids = tokenizer.encode(SAMPLE_TEXT)
        token_ids = token_ids[:4095] + token_ids[-1:]
        input_ids = torch.tensor(token_ids).unsqueeze(0)

        input_ids = input_ids.to(device=device)
        model = model.to(device=device, dtype=dtype)

        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        attention_mask[:, [1, 4, 21, ]] = 2

        output = model(input_ids, attention_mask=attention_mask)[0]
        output = output.float().sum()

        expected_output_sum = torch.tensor(76193.671875, device=device)  # with no padding needed, and fixed roberta-tokenizer

        print(f'device: {device}, dtype: {dtype}, attention_mode: {attention_mode} '
              f'Expected: {expected_output_sum}, Given: {output.sum()}')
        atol = 1e-2 if dtype == torch.half else 1e-4
        self.assertTrue(torch.allclose(output.sum(), expected_output_sum, atol=atol))

    def test_outout(self):
        self._run_test('cpu', torch.float, 'sliding_chunks')
        self._run_test('cuda', torch.float, 'sliding_chunks')
        self._run_test('cuda', torch.float, 'tvm')

        # self._run_test('cuda', torch.half, 'sliding_chunks')
        # self._run_test('cuda', torch.half, 'tvm')


if __name__ == '__main__':
    unittest.main()
