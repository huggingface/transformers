import unittest

from transformers import LwDetrViTConfig, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...test_backbone_common import BackboneTesterMixin
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch
    from torch import nn

    from transformers import LwDetrViTBackbone


class LwDetrVitModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        num_labels=3,
        num_channels=3,
        use_labels=True,
        is_training=True,
        image_size=256,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        window_block_indices=[1],
        out_indices=[0],
        num_windows=16,
        dropout_prob=0.0,
        attn_implementation="eager",
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.num_channels = num_channels
        self.use_labels = use_labels
        self.image_size = image_size

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.window_block_indices = window_block_indices
        self.out_indices = out_indices
        self.num_windows = num_windows
        self.dropout_prob = dropout_prob
        self.attn_implementation = attn_implementation

        self.is_training = is_training

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return LwDetrViTConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            window_block_indices=self.window_block_indices,
            out_indices=self.out_indices,
            num_windows=self.num_windows,
            hidden_dropout_prob=self.dropout_prob,
            attention_probs_dropout_prob=self.dropout_prob,
            attn_implementation=self.attn_implementation,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict

    def create_and_check_backbone(self, config, pixel_values, labels):
        model = LwDetrViTBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify hidden states
        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape),
            [
                self.batch_size,
                self.hidden_size,
                self.get_config().num_windows_side ** 2,
                self.get_config().num_windows_side ** 2,
            ],
        )

        # verify channels
        self.parent.assertEqual(len(model.channels), len(config.out_features))
        self.parent.assertListEqual(model.channels, [config.hidden_size])

        # verify backbone works with out_features=None
        config.out_features = None
        model = LwDetrViTBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape),
            [self.batch_size, config.hidden_size, config.patch_size, config.patch_size],
        )

        # verify channels
        self.parent.assertEqual(len(model.channels), 1)
        self.parent.assertListEqual(model.channels, [config.hidden_size])


@require_torch
class LwDetrViTBackboneTest(ModelTesterMixin, BackboneTesterMixin, unittest.TestCase):
    all_model_classes = (LwDetrViTBackbone,) if is_torch_available() else ()
    config_class = LwDetrViTConfig
    test_resize_embeddings = False
    test_torch_exportable = True

    def setUp(self):
        self.model_tester = LwDetrVitModelTester(self)

    def test_backbone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_backbone(*config_and_inputs)

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_attention_outputs(self):
        def check_attention_output(inputs_dict, config, model_class):
            config._attn_implementation = "eager"
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            attentions = outputs.attentions

            windowed_attentions = [attentions[i] for i in self.model_tester.window_block_indices]
            unwindowed_attentions = [attentions[i] for i in self.model_tester.out_indices]

            expected_windowed_attention_shape = [
                self.model_tester.batch_size * self.model_tester.num_windows,
                self.model_tester.num_attention_heads,
                self.model_tester.get_config().num_windows_side ** 2,
                self.model_tester.get_config().num_windows_side ** 2,
            ]

            expected_unwindowed_attention_shape = [
                self.model_tester.batch_size,
                self.model_tester.num_attention_heads,
                self.model_tester.image_size,
                self.model_tester.image_size,
            ]

            for i, attention in enumerate(windowed_attentions):
                self.assertListEqual(
                    list(attention.shape),
                    expected_windowed_attention_shape,
                )

            for i, attention in enumerate(unwindowed_attentions):
                self.assertListEqual(
                    list(attention.shape),
                    expected_unwindowed_attention_shape,
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            check_attention_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True

            check_attention_output(inputs_dict, config, model_class)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_stages = self.model_tester.num_hidden_layers
            self.assertEqual(len(hidden_states), expected_num_stages + 1)

            # VitDet's feature maps are of shape (batch_size, num_channels, height, width)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [
                    self.model_tester.hidden_size,
                    self.model_tester.hidden_size,
                ],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    # overwrite since LwDetrVitDet only supports retraining gradients of hidden states
    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = self.has_attentions

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs.feature_maps[0]

        # Encoder-/Decoder-only models
        hidden_states = outputs.hidden_states[0]
        hidden_states.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)
