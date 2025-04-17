# TODO Matt: Add this to a proper file later
from .test_modeling_common import ModelTesterMixin, GenerationTesterMixin, ids_tensor, torch_device, is_torch_available, require_torch
from .test_configuration_common import ConfigTester
from .test_pipeline_mixin import PipelineTesterMixin
import transformers
import unittest
import tempfile
if is_torch_available():
    import torch

class CausalLMModelTester:
    _allowed_model_classes = ("Base", "ForCausalLM", "ForSequenceClassification", "ForTokenClassification")
    config_class = None
    model_classes = None
    pipeline_model_mapping = None

    def _verify_model_attributes(self):
        if not isinstance(self.config_class, str):
            raise ValueError("You have inherited from CausalLMModelTester but did not set the config_class attribute.")
        if not isinstance(self.model_classes, dict):
            raise ValueError("You have inherited from CausalLMModelTester but did not set the model_classes attribute. "
                             "This should be a dictionary mapping model tasks to model classes , like "
                             "{'Base': MyModel, 'ForCausalLM': MyModelForCausalLM}")
        if unknown_tasks := set(self.model_classes.keys()) - set(self._allowed_model_classes):
            raise ValueError(f"Unknown tasks {unknown_tasks} in model_classes. Allowed tasks are: {self._allowed_model_classes}")
        if not isinstance(self.pipeline_model_mapping, dict):
            raise ValueError("You have inherited from CausalLMModelTester but did not set the pipeline_model_mapping attribute. "
                             "This should be a dictionary mapping pipeline tasks to model classes, like "
                             "{'feature-extraction': MyModel, 'text-classification': MyModelForSequenceClassification}")

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
            num_key_value_heads=2,
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
        self._verify_model_attributes()
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
        self.num_key_value_heads = num_key_value_heads
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
        self.head_dim = self.hidden_size // self.num_attention_heads

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

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
        return self.config_class(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            head_dim=self.head_dim,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = self.model_class(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

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

@require_torch
class CausalLMModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    test_headmasking = False
    test_pruning = False
    model_tester_class = None

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = GemmaForCausalLM if is_torch_available() else None

    def setUp(self):
        if self.model_tester_class is None:
            raise ValueError("You have inherited from CausalLMModelTest but did not set the model_tester_class attribute.")
        self.model_tester = self.model_tester_class(self)
        self.config_tester = ConfigTester(self, config_class=self.model_tester.config_class)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_Gemma_sequence_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = self.model_tester.for_sequence_class(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Gemma_sequence_classification_model_for_single_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = self.model_tester.for_sequence_class(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Gemma_sequence_classification_model_for_multi_label(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = self.model_tester.for_sequence_class(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_Gemma_token_classification_model(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        token_labels = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.num_labels)
        model = self.model_tester.for_token_class(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=token_labels)
        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_labels),
        )

    @unittest.skip(reason="Gemma uses GQA on all models so the KV cache is a non standard format")
    def test_past_key_values_format(self):
        pass

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        self.skipTest(reason="Gemma flash attention does not support right padding")

    @require_torch_sdpa
    @require_torch_accelerator
    @slow
    def test_sdpa_equivalence(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_sdpa:
                self.skipTest(reason="Model does not support SDPA")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.float16, attn_implementation="sdpa"
                )
                model_sdpa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16,
                                                    attn_implementation="eager")
                model.to(torch_device)

                dummy_input = inputs_dict[model_class.main_input_name]
                dummy_input = dummy_input.to(torch_device)
                outputs = model(dummy_input, output_hidden_states=True)
                outputs_sdpa = model_sdpa(dummy_input, output_hidden_states=True)

                logits = outputs.hidden_states[-1]
                logits_sdpa = outputs_sdpa.hidden_states[-1]

                # gemma sdpa needs a high tolerance
                assert torch.allclose(logits_sdpa, logits, atol=3e-3)

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @is_flaky()
    @slow
    def test_flash_attn_2_equivalence(self):
        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn_2:
                self.skipTest(reason="Model does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, torch_dtype=torch.float16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, torch_dtype=torch.float16,
                                                    attn_implementation="eager")
                model.to(torch_device)

                dummy_input = inputs_dict[model_class.main_input_name]
                dummy_input = dummy_input.to(torch_device)
                outputs = model(dummy_input, output_hidden_states=True)
                outputs_fa = model_fa(dummy_input, output_hidden_states=True)

                logits = outputs.hidden_states[-1]
                logits_fa = outputs_fa.hidden_states[-1]

                # gemma flash attention 2 needs a high tolerance
                assert torch.allclose(logits_fa, logits, atol=3e-3)