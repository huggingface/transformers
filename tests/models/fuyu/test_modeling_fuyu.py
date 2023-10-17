from transformers import AutoTokenizer, FuyuConfig, FuyuModel, FuyuForCausalLM, is_torch_available, is_vision_available
from transformers.models.fuyu.processing_fuyu import FuyuProcessor
from transformers.models.fuyu.image_processing_fuyu import FuyuImageProcessor

import unittest
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


from transformers.testing_utils import (
    require_torch_gpu,
    slow,
    torch_device
)

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
        pretrained_model_name = 'huggingface/pre_release_model'
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        image_processor = FuyuImageProcessor()

        processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)
        text_prompt = "Generate a coco-style caption.\\n"

        image_path = "/fsx/pablo/adept-collab/adept-mm/mm-inference-for-hf/bus.png"
        image_pil = Image.open(image_path)

        self.model_inputs = processor(text=text_prompt, images=[image_pil])

        self.model_config = FuyuConfig()
        self.model = FuyuModel(self.model_config).from_pretrained(pretrained_model_name)

    def test_fuyu_processing(self):
        """
        Test to ensure that the standard processing on a gold example matches adept's code.
        """
        torch.testing.assert_allclose(self.model_inputs["image_patch_input_indices"], torch.Tensor([[
            0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
            14,  15,  16,  17,  18,  19,  20,  21,  -1,  22,  23,  24,  25,  26,
            27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,
            41,  42,  43,  -1,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
            54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  -1,  66,
            67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
            81,  82,  83,  84,  85,  86,  87,  -1,  88,  89,  90,  91,  92,  93,
            94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107,
            108, 109,  -1, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
            121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,  -1, 132, 133,
            134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
            148, 149, 150, 151, 152, 153,  -1, 154, 155, 156, 157, 158, 159, 160,
            161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174,
            175,  -1, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
            188, 189, 190, 191, 192, 193, 194, 195, 196, 197,  -1, 198, 199, 200,
            201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214,
            215, 216, 217, 218, 219,  -1, 220, 221, 222, 223, 224, 225, 226, 227,
            228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
            -1, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
            255, 256, 257, 258, 259, 260, 261, 262, 263,  -1, 264, 265, 266, 267,
            268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281,
            282, 283, 284, 285,  -1, 286, 287, 288, 289, 290, 291, 292, 293, 294,
            295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307,  -1,
            -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
            -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]]), atol=0)

        torch.testing.assert_allclose(self.model_inputs['image_padded_unpacked_tokens_tensor'], torch.Tensor([[
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71019,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71019,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71019,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71019,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71019,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71019,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71019,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71019,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71019,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71019,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71019,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71019,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71019,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71019,      1, 128340,
            71374,  71389, 120412,  71377,  71835,  71374,  73615,  71375,  71399,
            71435,  71122,  71013,  71013,  71013,  71013,  71013,  71013,  71013,
            71013,  71013,  71013]]), atol=0)

    def test_model_embeddings_match_adept(self):
        """
        This test is very slow and needs about 30GB of RAM to be run on the Fuyu-8b model. 
        """
        word_embeddings = self.model.embed_tokens(self.model_inputs['image_padded_unpacked_tokens_tensor'][0][None, :])
        expected_word_embedding_start = torch.Tensor([2.8908e-06, -1.4961e-05, -2.9564e-05, -1.4901e-05, -2.5153e-05,
                                                      -1.9312e-05, -2.5511e-05,  1.9431e-05, -7.5698e-06, -3.5286e-05])
        expected_word_embedding_end = torch.Tensor([8.8811e-06, -3.7909e-05, -4.8637e-05,  2.1100e-05, -1.4544e-05,
                                                    -3.0756e-05,  4.3511e-06, -1.5080e-05,  2.5153e-05])
        torch.testing.assert_allclose(word_embeddings[0][0][0:10], expected_word_embedding_start, atol=1e-7)
        torch.testing.assert_allclose(word_embeddings[0][0][-9:], expected_word_embedding_end, atol=1e-7)
        torch.testing.assert_allclose(word_embeddings.shape, torch.Size([1, 335, 4096]), atol=0)

        continuous_embeddings = self.model.vision_embed_tokens(self.model_inputs['image_patches'][0][0]).unsqueeze(0)

        expected_continuous_embedding_start = torch.Tensor([-0.2891,  0.0649, -0.0175,  0.1641, -0.4844,
                                                            -0.9062,  0.4473,  0.2412, -0.2461, -0.0430])
        expected_continuous_embedding_end = torch.Tensor([-0.2754, -0.1836,  0.2422, -0.3711,  0.0564,
                                                          -0.1099,  0.0378,  0.1367, -0.2100])
        torch.testing.assert_allclose(continuous_embeddings[0][0][0:10], expected_continuous_embedding_start, atol=1e-4)
        torch.testing.assert_allclose(continuous_embeddings[0][0][-9:], expected_continuous_embedding_end, atol=1e-4)

        torch.testing.assert_allclose(continuous_embeddings[0].shape, torch.Size([308, 4096]), atol=0)
