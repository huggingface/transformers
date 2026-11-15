import unittest
from types import SimpleNamespace

from transformers.pipelines import AnyToAnyPipeline, ImageTextToTextPipeline


class MultimodalGenerationKwargsTests(unittest.TestCase):
    def make_pipeline(self, pipeline_class):
        pipe = object.__new__(pipeline_class)
        pipe.processor = SimpleNamespace(tokenizer=object())
        pipe.model = SimpleNamespace(output_modalities=["text", "image", "audio"])
        pipe._postprocess_params = {}
        return pipe

    def check_max_new_tokens(self, pipeline_class):
        pipe = self.make_pipeline(pipeline_class)
        shared = {"do_sample": False}

        _, first, _ = pipe._sanitize_parameters(
            generate_kwargs=shared,
            max_new_tokens=2,
        )
        self.assertEqual(first["generate_kwargs"]["max_new_tokens"], 2)
        self.assertEqual(shared, {"do_sample": False})

        _, second, _ = pipe._sanitize_parameters(
            generate_kwargs=shared,
            max_new_tokens=3,
        )
        self.assertEqual(second["generate_kwargs"]["max_new_tokens"], 3)
        self.assertEqual(shared, {"do_sample": False})

    def check_stop_sequence(self, pipeline_class):
        pipe = self.make_pipeline(pipeline_class)
        shared = {"do_sample": False}

        _, first, _ = pipe._sanitize_parameters(
            generate_kwargs=shared,
            stop_sequence="STOP",
        )
        self.assertEqual(first["generate_kwargs"]["stop_strings"], ["STOP"])
        self.assertEqual(shared, {"do_sample": False})

        _, second, _ = pipe._sanitize_parameters(
            generate_kwargs=shared,
        )
        self.assertNotIn("stop_strings", second["generate_kwargs"])

    def test_image_text_to_text_max_new_tokens(self):
        self.check_max_new_tokens(ImageTextToTextPipeline)

    def test_any_to_any_max_new_tokens(self):
        self.check_max_new_tokens(AnyToAnyPipeline)

    def test_image_text_to_text_stop_sequence(self):
        self.check_stop_sequence(ImageTextToTextPipeline)

    def test_any_to_any_stop_sequence(self):
        self.check_stop_sequence(AnyToAnyPipeline)

    def test_any_to_any_generation_mode(self):
        pipe = self.make_pipeline(AnyToAnyPipeline)
        shared = {"do_sample": False}

        _, first, _ = pipe._sanitize_parameters(
            generate_kwargs=shared,
            generation_mode="image",
        )
        self.assertEqual(first["generate_kwargs"]["generation_mode"], "image")
        self.assertEqual(shared, {"do_sample": False})

        _, second, postprocess = pipe._sanitize_parameters(
            generate_kwargs=shared,
            generation_mode="text",
        )
        self.assertNotIn("generation_mode", second["generate_kwargs"])
        self.assertEqual(postprocess["generation_mode"], "text")

    def test_explicit_duplicate_still_raises(self):
        for cls in (ImageTextToTextPipeline, AnyToAnyPipeline):
            with self.subTest(pipeline=cls.__name__):
                pipe = self.make_pipeline(cls)
                with self.assertRaisesRegex(ValueError, "max_new_tokens"):
                    pipe._sanitize_parameters(
                        generate_kwargs={"max_new_tokens": 2},
                        max_new_tokens=3,
                    )
