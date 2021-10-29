import unittest

import numpy as np

from transformers import is_ov_available
from transformers.testing_utils import require_ov, require_tf, require_torch, slow


if is_ov_available():
    from transformers import (
        GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
        AutoTokenizer,
        OVAutoModel,
        OVAutoModelForMaskedLM,
        OVAutoModelForQuestionAnswering,
        OVAutoModelWithLMHead,
    )


@require_ov
class OVAutoModelIntegrationTest(unittest.TestCase):
    def test_download(self):
        OVAutoModel.from_pretrained("dkurt/test_openvino")


@require_ov
class OVBertForQuestionAnsweringTest(unittest.TestCase):
    def check_model(self, model, tok):
        context = """
        Soon her eye fell on a little glass box that
        was lying under the table: she opened it, and
        found in it a very small cake, on which the
        words “EAT ME” were beautifully marked in
        currants. “Well, I’ll eat it,” said Alice, “ and if
        it makes me grow larger, I can reach the key ;
        and if it makes me grow smaller, I can creep
        under the door; so either way I’ll get into the
        garden, and I don’t care which happens !”
        """

        question = "Where Alice should go?"

        # For better OpenVINO efficiency it's recommended to use fixed input shape.
        # So pad input_ids up to specific max_length.
        input_ids = tok.encode(
            question + " " + tok.sep_token + " " + context, return_tensors="pt", max_length=128, padding="max_length"
        )

        outputs = model(input_ids)

        start_pos = outputs.start_logits.argmax()
        end_pos = outputs.end_logits.argmax() + 1

        answer_ids = input_ids[0, start_pos:end_pos]
        answer = tok.convert_tokens_to_string(tok.convert_ids_to_tokens(answer_ids))

        self.assertEqual(answer, "the garden")

    def test_from_pt(self):
        tok = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        model = OVAutoModelForQuestionAnswering.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad", from_pt=True
        )
        self.check_model(model, tok)

    def test_from_ir(self):
        tok = AutoTokenizer.from_pretrained("dkurt/bert-large-uncased-whole-word-masking-squad-int8-0001")
        model = OVAutoModelForQuestionAnswering.from_pretrained(
            "dkurt/bert-large-uncased-whole-word-masking-squad-int8-0001"
        )
        self.check_model(model, tok)


@require_ov
@require_torch
class GPT2ModelTest(unittest.TestCase):
    @slow
    def test_model_from_pretrained(self):
        for model_name in GPT2_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = OVAutoModel.from_pretrained(model_name, from_pt=True)
            self.assertIsNotNone(model)

            input_ids = np.random.randint(0, 255, (1, 6))
            attention_mask = np.random.randint(0, 2, (1, 6))

            expected_shape = (1, 6, 768)
            output = model(input_ids, attention_mask=attention_mask)[0]
            self.assertEqual(output.shape, expected_shape)


@require_ov
@require_torch
class OVAlbertModelIntegrationTest(unittest.TestCase):
    def test_inference_no_head_absolute_embedding(self):
        model = OVAutoModel.from_pretrained("albert-base-v2", from_pt=True)
        input_ids = np.array([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = (1, 11, 768)
        self.assertEqual(output.shape, expected_shape)
        expected_slice = np.array(
            [[[-0.6513, 1.5035, -0.2766], [-0.6515, 1.5046, -0.2780], [-0.6512, 1.5049, -0.2784]]]
        )

        self.assertTrue(np.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))


@require_ov
@require_torch
class OVOPENAIGPTModelLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_openai_gpt(self):
        model = OVAutoModelWithLMHead.from_pretrained("openai-gpt", from_pt=True)
        input_ids = np.array([[481, 4735, 544]], dtype=np.int64)  # the president is
        expected_output_ids = [
            481,
            4735,
            544,
            246,
            963,
            870,
            762,
            239,
            244,
            40477,
            244,
            249,
            719,
            881,
            487,
            544,
            240,
            244,
            603,
            481,
        ]  # the president is a very good man. " \n " i\'m sure he is, " said the

        output_ids = model.generate(input_ids, do_sample=False)
        self.assertListEqual(output_ids[0].tolist(), expected_output_ids)


@require_ov
@require_torch
class RobertaModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = OVAutoModelForMaskedLM.from_pretrained("roberta-base", from_pt=True)

        input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        output = model(input_ids)[0]
        expected_shape = (1, 11, 50265)
        self.assertEqual(output.shape, expected_shape)
        # compare the actual values for a slice.
        expected_slice = np.array(
            [[[33.8802, -4.3103, 22.7761], [4.6539, -2.8098, 13.6253], [1.8228, -3.6898, 8.8600]]]
        )

        # roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
        # roberta.eval()
        # expected_slice = roberta.model.forward(input_ids)[0][:, :3, :3].detach()
        self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))

    @slow
    def test_inference_no_head(self):
        model = OVAutoModel.from_pretrained("roberta-base", from_pt=True)

        input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        output = model(input_ids)[0]
        # compare the actual values for a slice.
        expected_slice = np.array([[[-0.0231, 0.0782, 0.0074], [-0.1854, 0.0540, -0.0175], [0.0548, 0.0799, 0.1687]]])

        # roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
        # roberta.eval()
        # expected_slice = roberta.extract_features(input_ids)[:, :3, :3].detach()

        self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))


@require_ov
@require_tf
class TFRobertaModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_masked_lm(self):
        model = OVAutoModelForMaskedLM.from_pretrained("roberta-base", from_tf=True)

        input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        output = model(input_ids)[0]
        expected_shape = [1, 11, 50265]
        self.assertEqual(list(output.shape), expected_shape)
        # compare the actual values for a slice.
        expected_slice = np.array(
            [[[33.8802, -4.3103, 22.7761], [4.6539, -2.8098, 13.6253], [1.8228, -3.6898, 8.8600]]]
        )
        self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))

    @slow
    def test_inference_no_head(self):
        model = OVAutoModel.from_pretrained("roberta-base", from_tf=True)

        input_ids = np.array([[0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2]])
        output = model(input_ids)[0]
        # compare the actual values for a slice.
        expected_slice = np.array([[[-0.0231, 0.0782, 0.0074], [-0.1854, 0.0540, -0.0175], [0.0548, 0.0799, 0.1687]]])
        self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))


@require_ov
@require_tf
class OVTFDistilBertModelIntegrationTest(unittest.TestCase):
    def test_inference_masked_lm(self):
        model = OVAutoModel.from_pretrained("distilbert-base-uncased", from_tf=True)
        input_ids = np.array([[0, 1, 2, 3, 4, 5]])
        output = model(input_ids)[0]

        expected_shape = (1, 6, 768)
        self.assertEqual(output.shape, expected_shape)

        expected_slice = np.array(
            [
                [
                    [0.19261885, -0.13732955, 0.4119799],
                    [0.22150156, -0.07422661, 0.39037204],
                    [0.22756018, -0.0896414, 0.3701467],
                ]
            ]
        )
        self.assertTrue(np.allclose(output[:, :3, :3], expected_slice, atol=1e-4))


@require_ov
@require_torch
class OVDistilBertModelIntegrationTest(unittest.TestCase):
    def test_inference_no_head_absolute_embedding(self):
        model = OVAutoModel.from_pretrained("distilbert-base-uncased", from_pt=True)
        model.to(device="CPU")
        model.set_config(config={"CPU_BIND_THREAD": "YES"})

        input_ids = np.array([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 1588, 2]])
        attention_mask = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = (1, 11, 768)
        self.assertEqual(output.shape, expected_shape)
        expected_slice = np.array([[[-0.1639, 0.3299, 0.1648], [-0.1746, 0.3289, 0.1710], [-0.1884, 0.3357, 0.1810]]])

        self.assertTrue(np.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))
