import unittest

from transformers import AutoTokenizer, FuyuConfig, FuyuForCausalLM, FuyuModel, is_torch_available, is_vision_available
from transformers.models.fuyu.image_processing_fuyu import FuyuImageProcessor
from transformers.models.fuyu.processing_fuyu import FuyuProcessor
from transformers.testing_utils import require_torch_gpu, slow, torch_device

from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin

import requests
import io

if is_vision_available():
    from PIL import Image


if is_torch_available():
    import torch


# Copied from transformers.tests.llama.test_modelling_llama.LlamaModelTest with Llama->Fuyu
class FuyuModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        image_size=300,
        patch_size=30,
        num_channels=3,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return FuyuConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = FuyuModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = FuyuModel(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, attention_mask=input_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        model = FuyuForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.is_decoder = True
        config.add_cross_attention = True
        model = FuyuForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # first forward pass
        outputs = model(
            input_ids,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_mask = ids_tensor((self.batch_size, 3), vocab_size=2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([input_mask, next_mask], dim=-1)

        output_from_no_past = model(
            next_input_ids,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )["hidden_states"][0]
        output_from_past = model(
            next_tokens,
            attention_mask=next_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
        )["hidden_states"][0]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch_gpu
@slow
class FuyuIntegrationTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Currently, all these tests depend on a value of max_tokens_to_generate of 10.
    """

    def setUp(self):
        self.pretrained_model_name = "huggingface/new_model_release_weights"
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        image_processor = FuyuImageProcessor()

        self.processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)
        self.model = FuyuForCausalLM.from_pretrained(self.pretrained_model_name)

        self.bus_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bus.png"
        self.bus_image_pil = Image.open(io.BytesIO(requests.get(self.bus_image_url).content))

    def test_model_embeddings_match_adept(self):
        """
        This test is very slow and needs about 30GB of RAM to be run on the Fuyu-8b model.
        """
        text_prompt_coco_captioning = "Generate a coco-style caption.\n"

        model_inputs_bus_captioning = self.processor(text=text_prompt_coco_captioning, images=self.bus_image_pil)

        continuous_embeddings = self.model.model.vision_embed_tokens(
            model_inputs_bus_captioning['image_patches']).unsqueeze(0)
        EXPECTED_CONTINUOUS_EMBEDDING_START = torch.Tensor(
            [-0.1221, 0.1689, -0.2969, 0.0601, 0.2168, -0.6953, 0.3438, 0.0165, 0.2168, 0.0586]
        )
        EXPECTED_CONTINUOUS_EMBEDDING_END = torch.Tensor(
            [0.1138, 0.2090, -0.0588, 0.0400, 0.1719, 0.0586, 0.0928, -0.1875, 0.0471]
        )
        torch.testing.assert_close(continuous_embeddings[0].shape, torch.Size([308, 4096]))
        torch.testing.assert_close(
            continuous_embeddings[0][0][0:10], EXPECTED_CONTINUOUS_EMBEDDING_START, rtol=0.1, atol=1e-02
        )
        torch.testing.assert_close(
            continuous_embeddings[0][0][-9:], EXPECTED_CONTINUOUS_EMBEDDING_END, rtol=0.1, atol=1e-02
        )

        # word_embeddings = self.model.embed_tokens(self.model_inputs['image_padded_unpacked_tokens_tensor'][0][None, :])
        word_embeddings = self.model.model.embed_tokens(model_inputs_bus_captioning['input_ids'])

        # fmt: off
        EXPECTED_WORD_EMBEDDING_START = torch.Tensor([2.0117e-06,-1.0371e-05,-2.0504e-05,-1.0312e-05,-1.7405e-05,-1.3471e-05,-1.7643e-05,1.3530e-05,-5.2452e-06,-2.4557e-05])
        EXPECTED_WORD_EMBEDDING_END = torch.Tensor([-2.6345e-05, -3.3855e-05, 1.4663e-05, -1.0133e-05, -2.1338e-05, 3.0249e-06, -1.0490e-05, 1.7405e-05, -1.1250e-06])
        # fmt: on

        torch.testing.assert_close(word_embeddings.shape, torch.Size([1, 335, 4096]))
        torch.testing.assert_close(word_embeddings[0][0][0:10], EXPECTED_WORD_EMBEDDING_START, rtol=0.1, atol=1e-02)
        torch.testing.assert_close(word_embeddings[0][0][-9:], EXPECTED_WORD_EMBEDDING_END, rtol=0.1, atol=1e-02)

    def test_model_forward_values(self):
        # fmt: off
        EXPECTED_HIDDEN_STATES_SLICE = torch.Tensor([[-0.5469, 1.6016, 2.3438, 2.8125, 1.0000],[0.3613, 1.0391, 2.5625, 2.2031, 1.5703],[-0.4707, 2.0938, 1.7109, 5.7188, 0.4199],])
        # fmt: on
        text_prompt_coco_captioning = "Generate a coco-style caption.\n"

        model_inputs_bus_captioning = self.processor(text=text_prompt_coco_captioning, images=self.bus_image_pil)
        model_outputs = self.model.model(**model_inputs_bus_captioning)
        torch.testing.assert_close(
            model_outputs[0][0, 5:8, 1200:1205], EXPECTED_HIDDEN_STATES_SLICE, rtol=0.1, atol=0.1
        )

    @slow
    @require_torch_gpu
    def test_model_8b_chat_greedy_generation_bus_captioning(self):
        EXPECTED_TEXT_COMPLETION = """A bus parked on the side of a road.|ENDOFTEXT|"""
        text_prompt_coco_captioning = "Generate a coco-style caption.\n"

        model_inputs_bus_captioning = self.processor(text=text_prompt_coco_captioning, images=self.bus_image_pil)
        generated_tokens = self.model.generate(**model_inputs_bus_captioning, max_new_tokens=10)
        text = self.processor.tokenizer.batch_decode(generated_tokens)
        end_sequence = text[0].split("\x04")[1]
        self.assertEqual(EXPECTED_TEXT_COMPLETION, end_sequence)  # TODO replace with postprocessing

    @slow
    @require_torch_gpu
    def test_model_8b_chat_greedy_generation_bus_color(self):
        EXPECTED_TEXT_COMPLETION = """The bus is blue.\n|ENDOFTEXT|"""
        text_prompt_bus_color = "What color is the bus?\n"
        model_inputs_bus_color = self.processor(text=text_prompt_bus_color, images=self.bus_image_pil)

        text = self.model.generate(**model_inputs_bus_color, max_new_tokens=10)

        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)

    @slow
    @require_torch_gpu
    def test_model_8b_chat_greedy_generation_chart_vqa(self):
        EXPECTED_TEXT_TOKENS = ['The', 'life expectancy', 'at', 'birth', 'of male',
                                's in', '', '20', '18', 'is', '', '80', '.', '7', '.', '\n', '|ENDOFTEXT|']
        expected_text_completion = " ".join(EXPECTED_TEXT_TOKENS)  # TODO make sure the end string matches

        text_prompt_chart_vqa = "What is the highest life expectancy at birth of male?\n"

        chart_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/chart.png"
        chart_image_pil = Image.open(io.BytesIO(requests.get(chart_image_url).content))

        model_inputs_chart_vqa = self.processor(text=text_prompt_chart_vqa, images=chart_image_pil)
        text = self.model.generate(**model_inputs_chart_vqa, max_new_tokens=10)

        self.assertEqual(expected_text_completion, text)

    @slow
    @require_torch_gpu
    def test_model_8b_chat_greedy_generation_bounding_box(self):
        EXPECTED_TEXT_COMPLETION = """\x00194213202244\x01|ENDOFTEXT|"""
        text_prompt_bbox = """When presented with a box, perform OCR to extract text contained within it. If provided with text, generate the corresponding bounding box.\\nWilliams"""  # noqa: E231

        bbox_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bbox_sample_image.png"
        bbox_image_pil = Image.open(io.BytesIO(requests.get(bbox_image_url).content))

        model_inputs_bbox = self.processor(text=text_prompt_bbox, images=bbox_image_pil)
        text = self.model.generate(**model_inputs_bbox, max_new_tokens=10)
        self.assertEqual(EXPECTED_TEXT_COMPLETION, text)
