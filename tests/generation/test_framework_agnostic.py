"""
Framework agnostic tests for generate()-related methods.
"""

import numpy as np

from transformers import AutoTokenizer
from transformers.testing_utils import slow, torch_device


class GenerationIntegrationTestsMixin:
    # To be populated by the child classes
    framework_dependent_parameters = {
        "AutoModelForCausalLM": None,
        "AutoModelForSpeechSeq2Seq": None,
        "AutoModelForSeq2SeqLM": None,
        "AutoModelForVision2Seq": None,
        "LogitsProcessorList": None,
        "MinLengthLogitsProcessor": None,
        "create_tensor_fn": None,
        "floats_tensor": None,
        "return_tensors": None,
        "set_seed": None,
    }

    def test_validate_generation_inputs(self):
        model_cls = self.framework_dependent_parameters["AutoModelForSeq2SeqLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        create_tensor_fn = self.framework_dependent_parameters["create_tensor_fn"]

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
        model = model_cls.from_pretrained("hf-internal-testing/tiny-random-t5")

        encoder_input_str = "Hello world"
        input_ids = tokenizer(encoder_input_str, return_tensors=return_tensors).input_ids

        # typos are quickly detected (the correct argument is `do_sample`)
        with self.assertRaisesRegex(ValueError, "do_samples"):
            model.generate(input_ids, do_samples=True)

        # arbitrary arguments that will not be used anywhere are also not accepted
        with self.assertRaisesRegex(ValueError, "foo"):
            fake_model_kwargs = {"foo": "bar"}
            model.generate(input_ids, **fake_model_kwargs)

        # however, valid model_kwargs are accepted
        valid_model_kwargs = {"attention_mask": create_tensor_fn(np.zeros_like(input_ids))}
        model.generate(input_ids, **valid_model_kwargs)

    def test_custom_logits_processor(self):
        model_cls = self.framework_dependent_parameters["AutoModelForSeq2SeqLM"]
        logits_processor_list_cls = self.framework_dependent_parameters["LogitsProcessorList"]
        min_length_logits_processor_cls = self.framework_dependent_parameters["MinLengthLogitsProcessor"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]

        bart_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_model = model_cls.from_pretrained("hf-internal-testing/tiny-random-bart", min_length=1)
        input_ids = bart_tokenizer(article, return_tensors=return_tensors).input_ids

        logits_processor = logits_processor_list_cls()
        logits_processor.append(min_length_logits_processor_cls(min_length=10, eos_token_id=0))
        # it should not be allowed to both define `min_length` via config and `logits_processor` list
        with self.assertRaises(ValueError):
            bart_model.generate(input_ids, logits_processor=logits_processor)

        bart_model.config.min_length = None
        bart_model.generate(input_ids, logits_processor=logits_processor)

    def test_max_new_tokens_encoder_decoder(self):
        model_cls = self.framework_dependent_parameters["AutoModelForSeq2SeqLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        is_pt = not model_cls.__name__.startswith("TF")

        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")

        bart_model = model_cls.from_pretrained("hf-internal-testing/tiny-random-bart")
        input_ids = bart_tokenizer(article, return_tensors=return_tensors).input_ids
        if is_pt:
            bart_model = bart_model.to(torch_device)
            input_ids = input_ids.to(torch_device)

        self.assertEqual(list(input_ids.shape), [1, 29])

        max_new_tokens = 3
        bart_model.config.max_length = 20
        bart_model.config.eos_token_id = None

        # Encoder decoder call
        outputs = bart_model.generate(input_ids, max_new_tokens=max_new_tokens)
        # 1 BOS + 3 new tokens
        self.assertEqual(list(outputs.shape), [1, 4])

        # Decoder only call
        outputs = bart_model.generate(decoder_input_ids=input_ids, max_new_tokens=max_new_tokens)
        # 1 BOS + 29 (input length) + 3 new tokens
        self.assertEqual(list(outputs.shape), [1, 33])

        # Encoder decoder call > 20
        outputs = bart_model.generate(max_new_tokens=max_new_tokens + 20)

        # 1 BOS + 20 + 3 new tokens
        self.assertEqual(list(outputs.shape), [1, 24])

    def test_max_new_tokens_decoder_only(self):
        model_cls = self.framework_dependent_parameters["AutoModelForCausalLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        is_pt = not model_cls.__name__.startswith("TF")

        article = """Justin Timberlake."""
        gpt2_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        gpt2_model = model_cls.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        input_ids = gpt2_tokenizer(article, return_tensors=return_tensors).input_ids
        if is_pt:
            gpt2_model = gpt2_model.to(torch_device)
            input_ids = input_ids.to(torch_device)

        self.assertEqual(list(input_ids.shape), [1, 9])

        max_new_tokens = 3
        gpt2_model.config.max_length = 20

        # call < 20
        outputs = gpt2_model.generate(input_ids, max_new_tokens=max_new_tokens)

        # 9 input_ids + 3 new tokens
        self.assertEqual(list(outputs.shape), [1, 12])

        # call > 20
        outputs = gpt2_model.generate(max_new_tokens=max_new_tokens + 20)

        # 1 BOS token + 23 new tokens
        self.assertEqual(list(outputs.shape), [1, 24])

    def test_encoder_decoder_generate_with_inputs_embeds(self):
        model_cls = self.framework_dependent_parameters["AutoModelForSeq2SeqLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]

        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        model = model_cls.from_pretrained("hf-internal-testing/tiny-random-bart", max_length=5)
        model.config.eos_token_id = None
        input_ids = tokenizer(article, return_tensors=return_tensors).input_ids

        inputs_embeds = model.get_input_embeddings()(input_ids)

        output_sequences = model.generate(inputs_embeds=inputs_embeds)

        # make sure model generated correctly until `max_length`
        self.assertEqual(output_sequences.shape, (1, 5))

    def test_transition_scores_greedy_search(self):
        model_cls = self.framework_dependent_parameters["AutoModelForCausalLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        is_pt = not model_cls.__name__.startswith("TF")

        articles = ["Justin Timberlake", "Michael Phelps"]
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2", padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token

        model = model_cls.from_pretrained("distilbert/distilgpt2")
        input_ids = tokenizer(articles, return_tensors=return_tensors, padding=True).input_ids
        if is_pt:
            model = model.to(torch_device)
            input_ids = input_ids.to(torch_device)

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=5,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_scores=True,
        )

        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores)
        if is_pt:
            transition_scores = transition_scores.cpu().numpy()

        expected_scores = np.array(
            [
                [-57.8844, -60.45698, -70.16364, -65.50791, -66.35648],
                [-54.417572, -60.216614, -62.661243, -58.621933, -58.298683],
            ]
        )
        self.assertTrue(np.allclose(transition_scores, expected_scores, atol=1e-3))

    def test_transition_scores_greedy_search_normalized(self):
        model_cls = self.framework_dependent_parameters["AutoModelForCausalLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        is_pt = not model_cls.__name__.startswith("TF")

        articles = ["Justin Timberlake", "Michael Phelps"]
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2", padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token

        model = model_cls.from_pretrained("distilbert/distilgpt2")
        input_ids = tokenizer(articles, return_tensors=return_tensors, padding=True).input_ids
        if is_pt:
            model = model.to(torch_device)
            input_ids = input_ids.to(torch_device)

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=5,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_scores=True,
        )

        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        if is_pt:
            transition_scores = transition_scores.cpu().numpy()

        expected_scores = np.array(
            [
                [-2.538938, -2.2694316, -2.1580915, -1.572299, -2.6719835],
                [-1.8826028, -2.2461371, -1.7556462, -2.9644494, -1.7996008],
            ]
        )
        self.assertTrue(np.allclose(transition_scores, expected_scores, atol=1e-3))

    def test_transition_scores_beam_search_encoder_decoder(self):
        model_cls = self.framework_dependent_parameters["AutoModelForSeq2SeqLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        is_pt = not model_cls.__name__.startswith("TF")

        articles = [
            "Justin Timberlake and Jessica Biel, welcome to parenthood.",
            "Michael Phelps is arguably the most decorated Olympian of all time.",
        ]
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")

        model = model_cls.from_pretrained(
            "hf-internal-testing/tiny-random-bart",
            max_length=10,
            num_beams=4,
            num_return_sequences=2,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_scores=True,
            length_penalty=0.0,
        )
        input_ids = tokenizer(articles, return_tensors=return_tensors, padding=True).input_ids
        if is_pt:
            model = model.to(torch_device)
            input_ids = input_ids.to(torch_device)

        outputs = model.generate(input_ids=input_ids)

        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices)
        if is_pt:
            transition_scores = transition_scores.cpu().numpy()
            outputs.sequences_scores = outputs.sequences_scores.cpu().numpy()

        self.assertTrue(np.allclose(np.sum(transition_scores, axis=-1), outputs.sequences_scores, atol=1e-3))

    def test_transition_scores_beam_search_encoder_decoder_with_eos(self):
        model_cls = self.framework_dependent_parameters["AutoModelForSeq2SeqLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        is_pt = not model_cls.__name__.startswith("TF")

        articles = [
            "Justin Timberlake and Jessica Biel, welcome to parenthood.",
            "Michael Phelps is arguably the most decorated Olympian of all time.",
        ]
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")

        model = model_cls.from_pretrained(
            "hf-internal-testing/tiny-random-bart",
            max_length=10,
            num_beams=4,
            num_return_sequences=2,
            return_dict_in_generate=True,
            output_scores=True,
            length_penalty=0.0,
        )
        input_ids = tokenizer(articles, return_tensors=return_tensors, padding=True).input_ids
        if is_pt:
            model = model.to(torch_device)
            input_ids = input_ids.to(torch_device)

        outputs = model.generate(input_ids=input_ids)

        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices)
        if is_pt:
            transition_scores = transition_scores.cpu().numpy()
            outputs.sequences_scores = outputs.sequences_scores.cpu().numpy()

        self.assertTrue(np.allclose(np.sum(transition_scores, axis=-1), outputs.sequences_scores, atol=1e-3))

    def test_transition_scores_beam_search_decoder_only(self):
        model_cls = self.framework_dependent_parameters["AutoModelForCausalLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        is_pt = not model_cls.__name__.startswith("TF")

        articles = [
            "Justin Timberlake",
            "Michael Phelps",
        ]
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        model = model_cls.from_pretrained(
            "hf-internal-testing/tiny-random-gpt2",
            max_length=10,
            num_beams=4,
            num_return_sequences=2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_scores=True,
            length_penalty=0.0,
        )
        input_ids = tokenizer(articles, return_tensors=return_tensors, padding=True).input_ids
        if is_pt:
            model = model.to(torch_device)
            input_ids = input_ids.to(torch_device)

        outputs = model.generate(input_ids=input_ids)

        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices)
        if is_pt:
            transition_scores = transition_scores.cpu().numpy()
            outputs.sequences_scores = outputs.sequences_scores.cpu().numpy()

        self.assertTrue(np.allclose(np.sum(transition_scores, axis=-1), outputs.sequences_scores, atol=1e-3))

    def test_transition_scores_beam_sample_encoder_decoder(self):
        model_cls = self.framework_dependent_parameters["AutoModelForSeq2SeqLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        is_pt = not model_cls.__name__.startswith("TF")

        articles = [
            "Justin Timberlake and Jessica Biel, welcome to parenthood.",
            "Michael Phelps is arguably the most decorated Olympian of all time.",
        ]
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")

        model = model_cls.from_pretrained(
            "hf-internal-testing/tiny-random-bart",
            do_sample=True,
            max_length=10,
            num_beams=4,
            num_return_sequences=2,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_scores=True,
            length_penalty=0.0,
        )
        input_ids = tokenizer(articles, return_tensors=return_tensors, padding=True).input_ids
        if is_pt:
            model = model.to(torch_device)
            input_ids = input_ids.to(torch_device)

        outputs = model.generate(input_ids=input_ids)

        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, outputs.beam_indices)
        if is_pt:
            transition_scores = transition_scores.cpu().numpy()
            outputs.sequences_scores = outputs.sequences_scores.cpu().numpy()

        self.assertTrue(np.allclose(np.sum(transition_scores, axis=-1), outputs.sequences_scores, atol=1e-3))

    @slow
    def test_transition_scores_early_stopping(self):
        # This is an aggressive test that makes sure that `beam_search's`
        # transition scores are computed correctly for varying `num_return_sequences`, `num_beams` and `batch_size > 1`
        # 2 x input_ids for "question: How are you? \n context: I had a long day, "
        model_cls = self.framework_dependent_parameters["AutoModelForSeq2SeqLM"]
        create_tensor_fn = self.framework_dependent_parameters["create_tensor_fn"]
        is_pt = not model_cls.__name__.startswith("TF")

        input_ids = create_tensor_fn(2 * [[822, 10, 571, 33, 25, 58, 2625, 10, 27, 141, 3, 9, 307, 239, 6, 1]])
        model = model_cls.from_pretrained("google-t5/t5-small")
        if is_pt:
            model = model.to(torch_device)
            input_ids = input_ids.to(torch_device)

        outputs = model.generate(
            input_ids,
            max_length=10,
            return_dict_in_generate=True,
            output_scores=True,
            forced_eos_token_id=model.config.eos_token_id,
            num_beams=4,
            do_sample=False,
            num_return_sequences=3,
            length_penalty=0.0,
        )

        transition_scores = model.compute_transition_scores(
            sequences=outputs.sequences, scores=outputs.scores, beam_indices=outputs.beam_indices
        )
        if is_pt:
            transition_scores = transition_scores.cpu().numpy()
            outputs.sequences_scores = outputs.sequences_scores.cpu().numpy()

        self.assertTrue(np.allclose(np.sum(transition_scores, axis=-1), outputs.sequences_scores))

    def test_encoder_decoder_generate_attention_mask(self):
        model_cls = self.framework_dependent_parameters["AutoModelForSeq2SeqLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        is_pt = not model_cls.__name__.startswith("TF")

        articles = ["Timberlake", "Jessica Biel, welcome to parenthood among other things"]
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        # need extreme generation values here to force this test
        # to fail when `attention_mask` is not correctly treated in generate
        model = model_cls.from_pretrained(
            "hf-internal-testing/tiny-random-bart", max_length=50, num_beams=5, num_return_sequences=5
        )
        model.config.eos_token_id = None
        input_ids = tokenizer(articles[0], return_tensors=return_tensors).input_ids
        input_ids_batched = tokenizer(articles, padding=True, return_tensors=return_tensors).input_ids
        if is_pt:
            model = model.to(torch_device)
            input_ids = input_ids.to(torch_device)
            input_ids_batched = input_ids_batched.to(torch_device)

        output_sequences_batched = model.generate(
            input_ids=input_ids_batched, return_dict_in_generate=True, output_scores=True
        )
        output_sequences = model.generate(input_ids=input_ids, return_dict_in_generate=True, output_scores=True)

        batched_out = output_sequences_batched.sequences_scores
        out = output_sequences.sequences_scores
        if is_pt:
            batched_out = batched_out.cpu().numpy()
            out = out.cpu().numpy()

        diff = np.abs(np.sum(batched_out[:5]) - np.sum(out))
        self.assertTrue(diff < 1e-4)

    def test_generate_input_ids_as_kwarg(self):
        model_cls = self.framework_dependent_parameters["AutoModelForCausalLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        is_pt = not model_cls.__name__.startswith("TF")

        article = """I need input_ids to generate"""
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = model_cls.from_pretrained("hf-internal-testing/tiny-random-gpt2", max_length=15)
        input_ids = tokenizer(article, return_tensors=return_tensors).input_ids
        if is_pt:
            model = model.to(torch_device)
            input_ids = input_ids.to(torch_device)

        output_sequences_kwargs = model.generate(input_ids=input_ids)
        output_sequences = model.generate(input_ids)
        if is_pt:
            output_sequences_kwargs = output_sequences_kwargs.cpu().numpy()
            output_sequences = output_sequences.cpu().numpy()

        self.assertTrue(np.array_equal(output_sequences, output_sequences_kwargs))
        self.assertEqual(output_sequences.shape, (1, 15))

    def test_generate_input_ids_as_encoder_kwarg(self):
        model_cls = self.framework_dependent_parameters["AutoModelForSeq2SeqLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        is_pt = not model_cls.__name__.startswith("TF")

        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        model = model_cls.from_pretrained("hf-internal-testing/tiny-random-bart", max_length=5)
        model.config.eos_token_id = None
        input_ids = tokenizer(article, return_tensors=return_tensors).input_ids
        if is_pt:
            model = model.to(torch_device)
            input_ids = input_ids.to(torch_device)

        output_sequences_kwargs = model.generate(input_ids=input_ids)
        output_sequences = model.generate(input_ids)
        if is_pt:
            output_sequences_kwargs = output_sequences_kwargs.cpu().numpy()
            output_sequences = output_sequences.cpu().numpy()

        self.assertTrue(np.array_equal(output_sequences, output_sequences_kwargs))
        self.assertEqual(output_sequences.shape, (1, 5))

    def test_generate_inputs_and_encoder_kwargs(self):
        model_cls = self.framework_dependent_parameters["AutoModelForCausalLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]

        article = """I need input_ids to generate"""
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = model_cls.from_pretrained("hf-internal-testing/tiny-random-gpt2", max_length=10)
        input_ids = tokenizer(article, return_tensors=return_tensors).input_ids
        with self.assertRaises(ValueError):
            model.generate(input_ids, input_ids=input_ids)

    def test_generate_too_many_encoder_kwargs(self):
        model_cls = self.framework_dependent_parameters["AutoModelForSeq2SeqLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]

        article = """I need input_ids to generate"""
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        model = model_cls.from_pretrained("hf-internal-testing/tiny-random-bart", max_length=10)
        input_ids = tokenizer(article, return_tensors=return_tensors).input_ids
        with self.assertRaises(ValueError):
            model.generate(input_ids=input_ids, inputs_embeds=input_ids)

    def test_generate_input_features_as_encoder_kwarg(self):
        model_cls = self.framework_dependent_parameters["AutoModelForSpeechSeq2Seq"]
        floats_tensor = self.framework_dependent_parameters["floats_tensor"]
        is_pt = not model_cls.__name__.startswith("TF")

        input_features = floats_tensor((3, 80, 60))
        model = model_cls.from_pretrained("hf-internal-testing/tiny-random-WhisperForConditionalGeneration")
        if is_pt:
            input_features.to(torch_device)
            model = model.to(torch_device)

        output_sequences_kwargs = model.generate(input_features=input_features, max_length=5)
        output_sequences = model.generate(input_features, max_length=5)
        if is_pt:
            output_sequences_kwargs = output_sequences_kwargs.cpu().numpy()
            output_sequences = output_sequences.cpu().numpy()

        self.assertTrue(np.array_equal(output_sequences, output_sequences_kwargs))
        self.assertEqual(output_sequences.shape, (3, 5))

    def test_generate_pixel_values_as_encoder_kwarg(self):
        model_cls = self.framework_dependent_parameters["AutoModelForVision2Seq"]
        floats_tensor = self.framework_dependent_parameters["floats_tensor"]
        is_pt = not model_cls.__name__.startswith("TF")

        pixel_values = floats_tensor((2, 3, 30, 30))
        model = model_cls.from_pretrained("hf-internal-testing/tiny-random-VisionEncoderDecoderModel-vit-gpt2")
        model.generation_config.eos_token_id = None
        if is_pt:
            pixel_values = pixel_values.to(torch_device)
            model = model.to(torch_device)

        output_sequences_kwargs = model.generate(pixel_values=pixel_values, max_length=5)
        output_sequences = model.generate(pixel_values, max_length=5)
        if is_pt:
            output_sequences_kwargs = output_sequences_kwargs.cpu().numpy()
            output_sequences = output_sequences.cpu().numpy()

        self.assertTrue(np.array_equal(output_sequences, output_sequences_kwargs))
        self.assertEqual(output_sequences.shape, (2, 5))

    def test_generate_encoder_outputs_attention_mask(self):
        model_cls = self.framework_dependent_parameters["AutoModelForSpeechSeq2Seq"]
        floats_tensor = self.framework_dependent_parameters["floats_tensor"]
        create_tensor_fn = self.framework_dependent_parameters["create_tensor_fn"]
        is_pt = not model_cls.__name__.startswith("TF")

        input_features = floats_tensor((3, 80, 60))
        attention_mask = create_tensor_fn(np.ones(input_features.shape))
        model = model_cls.from_pretrained("hf-internal-testing/tiny-random-WhisperForConditionalGeneration")
        if is_pt:
            input_features = input_features.to(torch_device)
            attention_mask = attention_mask.to(torch_device)
            model = model.to(torch_device)

        encoder = model.get_encoder()
        encoder_outputs = encoder(input_features)

        output_sequences_no_mask = model.generate(encoder_outputs=encoder_outputs)
        output_sequences_with_mask = model.generate(encoder_outputs=encoder_outputs, attention_mask=attention_mask)
        if is_pt:
            output_sequences_no_mask = output_sequences_no_mask.cpu().numpy()
            output_sequences_with_mask = output_sequences_with_mask.cpu().numpy()

        self.assertTrue(np.array_equal(output_sequences_no_mask, output_sequences_with_mask))

    def test_eos_token_id_int_and_list_greedy_search(self):
        model_cls = self.framework_dependent_parameters["AutoModelForCausalLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        is_pt = not model_cls.__name__.startswith("TF")

        generation_kwargs = {
            "do_sample": False,
            "num_beams": 1,
        }
        expectation = 13

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        text = """Hello, my dog is cute and"""
        tokens = tokenizer(text, return_tensors=return_tensors)
        model = model_cls.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        if is_pt:
            model = model.to(torch_device)
            tokens = tokens.to(torch_device)

        eos_token_id = 873
        generated_tokens = model.generate(**tokens, eos_token_id=eos_token_id, **generation_kwargs)
        self.assertTrue(expectation == len(generated_tokens[0]))

        eos_token_id = [873, 198]
        generated_tokens = model.generate(**tokens, eos_token_id=eos_token_id, **generation_kwargs)
        self.assertTrue(expectation == len(generated_tokens[0]))

    def test_eos_token_id_int_and_list_contrastive_search(self):
        model_cls = self.framework_dependent_parameters["AutoModelForCausalLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        is_pt = not model_cls.__name__.startswith("TF")

        generation_kwargs = {
            "do_sample": False,
            "num_beams": 1,
            "penalty_alpha": 0.6,
            "top_k": 4,
        }
        expectation = 17

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        text = """Hello, my dog is cute and"""
        tokens = tokenizer(text, return_tensors=return_tensors)
        model = model_cls.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        if is_pt:
            model = model.to(torch_device)
            tokens = tokens.to(torch_device)

        eos_token_id = 225
        generated_tokens = model.generate(**tokens, eos_token_id=eos_token_id, **generation_kwargs)
        self.assertTrue(expectation == len(generated_tokens[0]))

        eos_token_id = [225, 198]
        generated_tokens = model.generate(**tokens, eos_token_id=eos_token_id, **generation_kwargs)
        self.assertTrue(expectation == len(generated_tokens[0]))

    def test_eos_token_id_int_and_list_beam_search(self):
        model_cls = self.framework_dependent_parameters["AutoModelForCausalLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        is_pt = not model_cls.__name__.startswith("TF")

        generation_kwargs = {
            "do_sample": False,
            "num_beams": 3,
        }
        expectation = 13

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        text = """Hello, my dog is cute and"""
        tokens = tokenizer(text, return_tensors=return_tensors)
        model = model_cls.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        if is_pt:
            model = model.to(torch_device)
            tokens = tokens.to(torch_device)

        eos_token_id = 873
        generated_tokens = model.generate(**tokens, eos_token_id=eos_token_id, **generation_kwargs)
        unpadded_correct_condition = expectation == len(generated_tokens[0])
        padded_correct_condition = expectation < len(generated_tokens[0]) and all(
            token == model.config.pad_token_id for token in generated_tokens[0][expectation:]
        )
        self.assertTrue(unpadded_correct_condition or padded_correct_condition)

        eos_token_id = [873, 198]
        generated_tokens = model.generate(**tokens, eos_token_id=eos_token_id, **generation_kwargs)
        unpadded_correct_condition = expectation == len(generated_tokens[0])
        padded_correct_condition = expectation < len(generated_tokens[0]) and all(
            token == model.config.pad_token_id for token in generated_tokens[0][expectation:]
        )
        self.assertTrue(unpadded_correct_condition or padded_correct_condition)

    def test_generate_vision2text_conditioning(self):
        model_cls = self.framework_dependent_parameters["AutoModelForVision2Seq"]
        floats_tensor = self.framework_dependent_parameters["floats_tensor"]
        create_tensor_fn = self.framework_dependent_parameters["create_tensor_fn"]
        is_pt = not model_cls.__name__.startswith("TF")

        pixel_values = floats_tensor((2, 3, 30, 30))
        conditioning_input = create_tensor_fn([[10], [10]])  # this should be the 2nd output token, after the BOS token
        model = model_cls.from_pretrained("hf-internal-testing/tiny-random-VisionEncoderDecoderModel-vit-gpt2")
        if is_pt:
            pixel_values = pixel_values.to(torch_device)
            model = model.to(torch_device)
            conditioning_input = conditioning_input.to(torch_device)

        # we can condition on decoder_input_ids (expected decoder input) and input_ids (which we pipe internally as
        # decoder_input_ids, if the encoder is not a model with text input)
        output_sequences_decoder_input_ids = model.generate(
            pixel_values, max_length=5, decoder_input_ids=conditioning_input
        )
        output_sequences_input_ids = model.generate(pixel_values, max_length=5, input_ids=conditioning_input)
        if is_pt:
            output_sequences_decoder_input_ids = output_sequences_decoder_input_ids.cpu().numpy()
            output_sequences_input_ids = output_sequences_input_ids.cpu().numpy()
            conditioning_input = conditioning_input.cpu().numpy()

        self.assertTrue(np.array_equal(output_sequences_decoder_input_ids, output_sequences_input_ids))
        self.assertTrue(np.array_equal(output_sequences_decoder_input_ids[:, 1:2], conditioning_input))
