import copy
import inspect
import tempfile
import unittest

from transformers import UnivNetGanConfig
from transformers.testing_utils import (
    is_torch_available,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
    torch_device,
)
from transformers.trainer_utils import set_seed
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import UnivNetGan


# TODO: Copied from SpeechT5HifiGanTester, change to be valid for UniVNetGan
class UnivNetGanTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
        num_mel_bins=20,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.num_mel_bins = num_mel_bins

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.seq_length, self.num_mel_bins], scale=1.0)
        config = self.get_config()
        return config, input_values

    def get_config(self):
        return UnivNetGanConfig(
            model_in_dim=self.num_mel_bins,
        )

    def create_and_check_model(self, config, input_values):
        model = UnivNetGan(config=config).to(torch_device).eval()
        result = model(input_values)
        self.parent.assertEqual(result.shape, (self.seq_length * 256,))

    def prepare_config_and_inputs_for_common(self):
        config, input_values = self.prepare_config_and_inputs()
        inputs_dict = {"spectrogram": input_values}
        return config, inputs_dict


# TODO: Copied from SpeechT5HifiGanTest, change to be valid for UnivNetGan
@require_torch
class UnivNetGanTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (UnivNetGan,) if is_torch_available() else ()
    test_torchscript = False
    test_pruning = False
    test_resize_embeddings = False
    test_resize_position_embeddings = False
    test_head_masking = False
    test_mismatched_shapes = False
    test_missing_keys = False
    test_model_parallel = False
    is_encoder_decoder = False
    has_attentions = False

    input_name = "spectrogram"

    def setUp(self):
        self.model_tester = UnivNetGanTester(self)
        self.config_tester = ConfigTester(self, config_class=UnivNetGanConfig)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_from_and_save_pretrained_subfolder()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = [
                "spectrogram",
            ]
            self.assertListEqual(arg_names[: len(expected_arg_names)], expected_arg_names)

    # this model does not output hidden states
    def test_hidden_states_output(self):
        pass

    # skip
    def test_initialization(self):
        pass

    # this model has no inputs_embeds
    def test_inputs_embeds(self):
        pass

    # this model has no input embeddings
    def test_model_common_attributes(self):
        pass

    # skip as this model doesn't support all arguments tested
    def test_model_outputs_equivalence(self):
        pass

    # this model does not output hidden states
    def test_retain_grad_hidden_states_attentions(self):
        pass

    # skip because it fails on automapping of SpeechT5HifiGanConfig
    def test_save_load_fast_init_from_base(self):
        pass

    # skip because it fails on automapping of SpeechT5HifiGanConfig
    def test_save_load_fast_init_to_base(self):
        pass

    def test_batched_inputs_outputs(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            batched_inputs = inputs["spectrogram"].unsqueeze(0).repeat(2, 1, 1)
            with torch.no_grad():
                batched_outputs = model(batched_inputs.to(torch_device))

            self.assertEqual(
                batched_inputs.shape[0], batched_outputs.shape[0], msg="Got different batch dims for input and output"
            )

    def test_unbatched_inputs_outputs(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(inputs["spectrogram"].to(torch_device))
            self.assertTrue(outputs.dim() == 1, msg="Got un-batched inputs but batched output")