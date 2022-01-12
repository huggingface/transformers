from transformers import PerceiverConfig
from .test_modeling_tf_common import ids_tensor, floats_tensor, random_attention_mask, TFModelTesterMixin
from .test_configuration_common import ConfigTester
from transformers.testing_utils import require_tf, slow
import tensorflow as tf
from transformers.file_utils import is_tf_available
import unittest
import copy
from transformers.models.auto import get_values
import inspect
import numpy as np
from typing import List, Tuple, Dict
import tempfile


if is_tf_available():
    import tensorflow as tf

    from transformers import (
        TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
        TF_MODEL_FOR_MASKED_LM_MAPPING,
        TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        TF_MODEL_MAPPING,
        TFPerceiverForImageClassificationConvProcessing,
        TFPerceiverForImageClassificationFourier,
        TFPerceiverForImageClassificationLearned,
        TFPerceiverForMaskedLM,
        TFPerceiverForMultimodalAutoencoding,
        TFPerceiverForOpticalFlow,
        TFPerceiverForSequenceClassification,
        TFPerceiverModel,
        # TFPerceiverTokenizer,
    )


class TFPerceiverModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        num_channels=3,
        image_size=32,
        train_size=[20, 20],
        num_frames=5,
        audio_samples_per_frame=200,
        samples_per_patch=20,
        nchunks=20,
        num_latents=10,
        d_latents=20,
        num_blocks=1,
        num_self_attends_per_block=2,
        num_self_attention_heads=1,
        num_cross_attention_heads=1,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        max_position_embeddings=7,
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.image_size = image_size
        self.train_size = train_size
        self.num_frames = num_frames
        self.audio_samples_per_frame = audio_samples_per_frame
        self.samples_per_patch = samples_per_patch
        self.nchunks = nchunks
        self.num_latents = num_latents
        self.d_latents = d_latents
        self.num_blocks = num_blocks
        self.num_self_attends_per_block = num_self_attends_per_block
        self.num_self_attention_heads = num_self_attention_heads
        self.num_cross_attention_heads = num_cross_attention_heads
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_act = hidden_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope
        # set subsampling for multimodal model (take first chunk)
        # image_chunk_size = np.prod((self.num_frames, self.image_size, self.image_size)) // self.nchunks
        # audio_chunk_size = self.num_frames * self.audio_samples_per_frame // self.samples_per_patch // self.nchunks
        # self.subsampling = {
        #     "image": torch.arange(0, image_chunk_size),
        #     "audio": torch.arange(0, audio_chunk_size),
        #     "label": None,
        # }

    def prepare_config_and_inputs(self, model_class=None):
        config = self.get_config()

        input_mask = None
        sequence_labels = None
        token_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.num_labels)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        if model_class is None or model_class.__name__ == "TFPerceiverModel":
            inputs = floats_tensor([self.batch_size, self.seq_length, config.d_model], self.vocab_size)
            return config, inputs, input_mask, sequence_labels, token_labels
        elif model_class.__name__ in ["TFPerceiverForMaskedLM", "TFPerceiverForSequenceClassification"]:
            inputs = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            # input mask is only relevant for text inputs
            if self.use_input_mask:
                input_mask = random_attention_mask([self.batch_size, self.seq_length])
        elif model_class.__name__ == "TFPerceiverForImageClassificationLearned":
            inputs = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        elif model_class.__name__ == "TFPerceiverForImageClassificationFourier":
            inputs = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        elif model_class.__name__ == "TFPerceiverForImageClassificationConvProcessing":
            inputs = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        elif model_class.__name__ == "TFPerceiverForOpticalFlow":
            inputs = floats_tensor([self.batch_size, 2, 27, self.train_size[0], self.train_size[1]])
        elif model_class.__name__ == "TFPerceiverForMultimodalAutoencoding":
            images = tf.random.normal(
                (
                    self.batch_size,
                    self.num_frames,
                    self.num_channels,
                    self.image_size,
                    self.image_size,
                )  # pytroch has device argument
            )
            audio = tf.random.normal((self.batch_size, self.num_frames * self.audio_samples_per_frame, 1))
            inputs = dict(image=images, audio=audio, label=tf.zeros((self.batch_size, self.num_labels)))
        else:
            raise ValueError(f"Model class {model_class} not supported")

        return config, inputs, input_mask, sequence_labels, token_labels

    def get_config(self):
        return PerceiverConfig(
            num_latents=self.num_latents,
            d_latents=self.d_latents,
            num_blocks=self.num_blocks,
            num_self_attends_per_block=self.num_self_attends_per_block,
            num_self_attention_heads=self.num_self_attention_heads,
            num_cross_attention_heads=self.num_cross_attention_heads,
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            max_position_embeddings=self.max_position_embeddings,
            image_size=self.image_size,
            train_size=self.train_size,
            num_frames=self.num_frames,
            audio_samples_per_frame=self.audio_samples_per_frame,
            samples_per_patch=self.samples_per_patch,
            num_labels=self.num_labels,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        # Byte level vocab
        config.vocab_size = 261
        config.max_position_embeddings = 40
        return config

    def create_and_check_for_masked_lm(self, config, inputs, input_mask, sequence_labels, token_labels):
        model = TFPerceiverForMaskedLM(config=config)
        result = model(inputs, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(self, config, inputs, input_mask, sequence_labels, token_labels):
        model = TFPerceiverForSequenceClassification(config=config)
        result = model(inputs, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_image_classification_learned(
        self, config, inputs, input_mask, sequence_labels, token_labels
    ):
        model = TFPerceiverForImageClassificationLearned(config=config)
        result = model(inputs, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_image_classification_fourier(
        self, config, inputs, input_mask, sequence_labels, token_labels
    ):
        model = TFPerceiverForImageClassificationFourier(config=config)
        result = model(inputs, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_image_classification_conv(self, config, inputs, input_mask, sequence_labels, token_labels):
        model = TFPerceiverForImageClassificationConvProcessing(config=config)
        result = model(inputs, attention_mask=input_mask, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, inputs, input_mask, sequence_labels, token_labels = config_and_inputs
        inputs_dict = {"inputs": inputs, "attention_mask": input_mask}
        return config, inputs_dict

    def prepare_config_and_inputs_for_model_class(self, model_class):
        config_and_inputs = self.prepare_config_and_inputs(model_class)
        config, inputs, input_mask, sequence_labels, token_labels = config_and_inputs
        inputs_dict = {"inputs": inputs, "attention_mask": input_mask}

        return config, inputs_dict


@require_tf
class PerceiverModelTest(TFModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            TFPerceiverModel,
            TFPerceiverForMaskedLM,
            TFPerceiverForImageClassificationLearned,
            TFPerceiverForImageClassificationFourier,
            # TFPerceiverForImageClassificationConvProcessing,
            # TFPerceiverForOpticalFlow,
            # TFPerceiverForMultimodalAutoencoding,
            TFPerceiverForSequenceClassification,
        )
        if is_tf_available()
        else ()
    )
    test_pruning = False
    test_head_masking = False

    maxDiff = None

    def setUp(self):
        self.model_tester = TFPerceiverModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PerceiverConfig, hidden_size=37)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)

        if model_class.__name__ == "PerceiverForMultimodalAutoencoding":
            inputs_dict["subsampled_output_points"] = self.model_tester.subsampling

        if return_labels:
            if model_class in [
                *get_values(TF_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING),
                *get_values(TF_MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING),
            ]:
                inputs_dict["labels"] = tf.zeros(self.model_tester.batch_size, dtype=tf.int64)
            elif model_class in [
                *get_values(TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING),
                *get_values(TF_MODEL_FOR_MASKED_LM_MAPPING),
            ]:
                inputs_dict["labels"] = tf.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=tf.int64
                )
        return inputs_dict

    def test_config(self):
        # we don't test common_properties and arguments_init as these don't apply for Perceiver
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(model_class=TFPerceiverForMaskedLM)
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(
            model_class=TFPerceiverForSequenceClassification
        )
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_image_classification_learned(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(
            model_class=TFPerceiverForImageClassificationLearned
        )
        self.model_tester.create_and_check_for_image_classification_learned(*config_and_inputs)

    def test_for_image_classification_fourier(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(
            model_class=TFPerceiverForImageClassificationFourier
        )
        self.model_tester.create_and_check_for_image_classification_fourier(*config_and_inputs)

    def test_for_image_classification_conv(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(
            model_class=TFPerceiverForImageClassificationConvProcessing
        )
        self.model_tester.create_and_check_for_image_classification_conv(*config_and_inputs)

    def test_model_common_attributes(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
            model = model_class(config)
            # we overwrite this, as the embeddings of Perceiver are an instance of nn.Parameter
            # and Perceiver doesn't support get_output_embeddings
            self.assertIsInstance(model.get_input_embeddings(), (tf.Variable))

    def test_forward_signature(self):
        for model_class in self.all_model_classes:
            config, _ = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)

            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["inputs"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_determinism(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)

            model = model_class(config)
            inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            first = model(**inputs_dict)[0]
            second = model(**inputs_dict)[0]

            if model_class.__name__ == "PerceiverForMultimodalAutoencoding":
                # model outputs a dictionary with logits per modality, let's verify each modality
                for modality in first.keys():
                    out_1 = first[modality].cpu().numpy()
                    out_2 = second[modality].cpu().numpy()
                    out_1 = out_1[~np.isnan(out_1)]
                    out_2 = out_2[~np.isnan(out_2)]
                    max_diff = np.amax(np.abs(out_1 - out_2))
                    self.assertLessEqual(max_diff, 1e-5)
            else:
                out_1 = first.cpu().numpy()
                out_2 = second.cpu().numpy()
                out_1 = out_1[~np.isnan(out_1)]
                out_2 = out_2[~np.isnan(out_2)]
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    def test_attention_outputs(self):
        seq_len = getattr(self.model_tester, "num_latents", None)

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
            config.return_dict = True

            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self_attentions = outputs.attentions
            cross_attentions = outputs.cross_attentions

            # check expected number of attentions depending on model class
            expected_num_self_attentions = self.model_tester.num_blocks * self.model_tester.num_self_attends_per_block
            if model.__class__.__name__ == "TFPerceiverModel":
                # we expect to have 2 cross-attentions, namely one in the PerceiverEncoder, and one in PerceiverBasicDecoder
                expected_num_cross_attentions = 1
            else:
                # we expect to have 2 cross-attentions, namely one in the PerceiverEncoder, and one in PerceiverBasicDecoder
                expected_num_cross_attentions = 2
            self.assertEqual(len(self_attentions), expected_num_self_attentions)
            self.assertEqual(len(cross_attentions), expected_num_cross_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self_attentions = outputs.attentions
            cross_attentions = outputs.cross_attentions
            self.assertEqual(len(self_attentions), expected_num_self_attentions)
            self.assertEqual(len(cross_attentions), expected_num_cross_attentions)

            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_self_attention_heads, seq_len, seq_len],
            )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(out_len + 1, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), expected_num_self_attentions)
            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_self_attention_heads, seq_len, seq_len],
            )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = self.model_tester.num_blocks * self.model_tester.num_self_attends_per_block + 1
            self.assertEqual(len(hidden_states), expected_num_layers)

            seq_length = self.model_tester.num_latents

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.d_latents],
            )

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)

            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_model_outputs_equivalence(self):
        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
            dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

            def recursive_check(tuple_object, dict_object):
                if isinstance(tuple_object, (List, Tuple)):
                    for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                        recursive_check(tuple_iterable_value, dict_iterable_value)
                elif isinstance(tuple_object, Dict):
                    for tuple_iterable_value, dict_iterable_value in zip(tuple_object.values(), dict_object.values()):
                        recursive_check(tuple_iterable_value, dict_iterable_value)
                elif tuple_object is None:
                    return
                else:
                    self.assertTrue(
                        tf.debugging.assert_near(
                            set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                        ),
                        msg=f"Tuple and dict output are not equal. Difference: {tf.math.reduce_max(tf.math.abs(tuple_object - dict_object))}. "
                        f"Tuple has `nan`: {tf.reduce_any(tf.math.is_nan(tuple_object))} and `inf`: {tf.reduce_any(tf.math.is_inf(tuple_object))}. "
                        f"Dict has `nan`: {tf.reduce_any(tf.math.is_nan(dict_object))} and `inf`: {tf.reduce_any(tf.math.is_inf(dict_object))}.",
                    )

            recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)

            model = model_class(config)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs)

            if model_class.__name__ not in ["TFPerceiverForOpticalFlow", "TFPerceiverForMultimodalAutoencoding"]:
                # optical flow + multimodal models don't support training for now
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)

            check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

            if model_class.__name__ not in ["TFPerceiverForOpticalFlow", "TFPerceiverForMultimodalAutoencoding"]:
                # optical flow + multimodal models don't support training for now
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_hidden_states": True})

            if model_class.__name__ not in ["TFPerceiverForOpticalFlow", "TFPerceiverForMultimodalAutoencoding"]:
                # optical flow + multimodal models don't support training for now
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs, {"output_attentions": True})

            if model_class.__name__ not in ["TFPerceiverForOpticalFlow", "TFPerceiverForMultimodalAutoencoding"]:
                # optical flow + multimodal models don't support training for now
                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(
                    model, tuple_inputs, dict_inputs, {"output_hidden_states": True, "output_attentions": True}
                )

    def test_save_load(self):
        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)

            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if model_class.__name__ == "TFPerceiverForMultimodalAutoencoding":
                for modality in outputs[0].keys():
                    out_2 = outputs[0][modality].numpy()
                    out_2[np.isnan(out_2)] = 0

                    with tempfile.TemporaryDirectory() as tmpdirname:
                        model.save_pretrained(tmpdirname)
                        model = model_class.from_pretrained(tmpdirname)
                        after_outputs = model(**self._prepare_for_class(inputs_dict, model_class))

                        # Make sure we don't have nans
                        out_1 = after_outputs[0][modality].numpy()
                        out_1[np.isnan(out_1)] = 0
                        max_diff = np.amax(np.abs(out_1 - out_2))
                        self.assertLessEqual(max_diff, 1e-5)

            else:
                out_2 = outputs[0].numpy()
                out_2[np.isnan(out_2)] = 0

                with tempfile.TemporaryDirectory() as tmpdirname:
                    model.save_pretrained(tmpdirname)
                    model = model_class.from_pretrained(tmpdirname)
                    after_outputs = model(**self._prepare_for_class(inputs_dict, model_class))

                    # Make sure we don't have nans
                    out_1 = after_outputs[0].numpy()
                    out_1[np.isnan(out_1)] = 0
                    max_diff = np.amax(np.abs(out_1 - out_2))
                    self.assertLessEqual(max_diff, 1e-5)
