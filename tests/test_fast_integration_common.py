# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Shared mixin for fast integration tests.

Fast integration tests run the full processor → model → forward/generate
pipeline using tiny Hub models from hf-tiny-v2. They do not assert on
output values; their purpose is to catch integration bugs between the
processor and model (wrong shapes, missing keys, dtype mismatches, etc.)
that unit tests with synthetic tensors miss.

Unlike ``@slow`` integration tests these use models that are only a few MB,
run entirely on CPU, and require no ``RUN_SLOW=1`` gate.
"""

import gc

import numpy as np

from transformers import AutoProcessor
from transformers.testing_utils import require_torch


# ---------------------------------------------------------------------------
# Shared fixture paths (relative to the repo root)
# ---------------------------------------------------------------------------

IMAGE_FIXTURE_PATH = "tests/fixtures/tests_samples/COCO/000000039769.png"

# Synthetic audio defaults (used when no audio fixture path is supplied)
_AUDIO_SAMPLING_RATE = 16_000
_AUDIO_DURATION_S = 1  # 1 second of silence


class FastIntegrationTestMixin:
    """
    Mixin providing fast integration tests for a model using a tiny Hub checkpoint.

    These tests exercise the full pipeline::

        processor(inputs) → model(**inputs) / model.generate(**inputs)

    without asserting on specific output values.  Any exception in this
    pipeline is a test failure.

    **Required class attributes**

    ``model_id`` (str)
        Hub repo ID of the tiny model, e.g.
        ``"hf-tiny-v2/tiny-random-LlamaForCausalLM"``.
    ``all_model_classes`` (tuple)
        The model class(es) to instantiate. Usually just one, e.g.
        ``(LlamaForCausalLM,)``.  Forward is tested for every class;
        generate is tested only for classes where ``can_generate()`` is
        ``True``.

    **Optional class attributes**

    ``input_modalities`` (tuple[str])
        Which input modalities to build for the processor call.  Supported
        values: ``"text"``, ``"image"``, ``"audio"``, ``"video"``.
        Defaults to ``("text",)``.
    ``text_input`` (str or None)
        Explicit text string to pass to the processor.  When ``None`` the
        mixin builds a default string, auto-prepending the model's
        ``image_token`` / ``video_token`` when those modalities are used.
    ``image_fixture_path`` (str or None)
        Path to an image file.  Falls back to the shared COCO PNG.
    ``audio_fixture_path`` (str or None)
        Path to an audio file readable by ``soundfile``.  Falls back to a
        1-second synthetic silence array at 16 kHz.
    ``video_fixture_path`` (str or None)
        Path to a video file.  Falls back to four copies of the image
        fixture used as frames. Override ``_load_video()`` to implement
        actual video decoding.
    ``processor_call_kwargs`` (dict)
        Extra keyword arguments forwarded to every ``processor(...)`` call.
    """

    # -----------------------------------------------------------------------
    # Required
    # -----------------------------------------------------------------------
    model_id = None
    all_model_classes = ()

    # -----------------------------------------------------------------------
    # Optional — subclass overrides
    # -----------------------------------------------------------------------
    input_modalities = ("text",)

    text_input = None
    image_fixture_path = None
    audio_fixture_path = None
    video_fixture_path = None
    processor_call_kwargs = {}

    _DEFAULT_TEXT_INPUT = "Hello, I am a language model and I"

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    @classmethod
    def setUpClass(cls):
        """Load the processor once for the whole test class.

        If the processor cannot be loaded (repo has no processor config, or
        any other error) ``cls.processor`` is set to ``None`` and each test
        method will skip itself.
        """
        cls.processor = None
        if not cls.model_id or not cls.all_model_classes:
            return
        try:
            cls.processor = AutoProcessor.from_pretrained(cls.model_id)
        except Exception:
            pass

    @classmethod
    def tearDownClass(cls):
        del cls.processor
        gc.collect()

    # -----------------------------------------------------------------------
    # Skip helper
    # -----------------------------------------------------------------------

    def _skip_if_no_processor(self):
        if self.processor is None:
            self.skipTest(
                f"Processor not available for '{self.model_id}' — "
                "either the repo has no processor config or loading failed. "
                "Skipping fast integration test."
            )

    # -----------------------------------------------------------------------
    # Input builders
    # -----------------------------------------------------------------------

    def _get_text(self):
        """Return the text string to pass to the processor.

        When ``text_input`` is not set the mixin builds a default string and
        auto-prepends the processor's ``image_token`` / ``video_token`` so
        that models that require special placeholder tokens in the text
        receive a valid input without the subclass having to hard-code token
        strings.
        """
        if self.text_input is not None:
            return self.text_input

        text = self._DEFAULT_TEXT_INPUT
        if self.processor is not None:
            if "video" in self.input_modalities:
                token = getattr(self.processor, "video_token", "") or ""
                if token:
                    text = f"{token}\n{text}"
            elif "image" in self.input_modalities:
                token = getattr(self.processor, "image_token", "") or ""
                if token:
                    text = f"{token}\n{text}"
        return text

    def _load_image(self):
        """Return a PIL image from the image fixture."""
        from PIL import Image

        return Image.open(self.image_fixture_path or IMAGE_FIXTURE_PATH)

    def _load_audio(self):
        """Return a dict of audio kwargs suitable for unpacking into the processor call.

        The default implementation produces a 1-second silence array at
        16 kHz.  Override this method when the model's processor uses
        non-standard kwarg names or needs a different format.
        """
        if self.audio_fixture_path:
            import soundfile as sf

            audio, sr = sf.read(self.audio_fixture_path)
            return {"audio": audio, "sampling_rate": sr}
        audio = np.zeros(_AUDIO_SAMPLING_RATE * _AUDIO_DURATION_S, dtype=np.float32)
        return {"audio": audio, "sampling_rate": _AUDIO_SAMPLING_RATE}

    def _load_video(self):
        """Return one video as a list of PIL image frames.

        The default implementation returns four copies of the image fixture.
        Override this method when ``video_fixture_path`` is set or when the
        model requires a specific frame format.
        """
        if self.video_fixture_path:
            raise NotImplementedError(
                "Override _load_video() to decode frames from self.video_fixture_path"
            )
        return [self._load_image()] * 4

    def _get_processor_inputs(self):
        """Build the kwargs dict to unpack into the processor call."""
        kwargs = {}
        if "text" in self.input_modalities:
            kwargs["text"] = self._get_text()
        if "image" in self.input_modalities:
            kwargs["images"] = self._load_image()
        if "audio" in self.input_modalities:
            kwargs.update(self._load_audio())
        if "video" in self.input_modalities:
            # Wrap in an outer list: the processor expects a *batch* of
            # videos, where each video is itself a list of frames.
            kwargs["videos"] = [self._load_video()]
        return kwargs

    def _prepare_model_inputs(self, model, inputs):
        """Optional post-processing of tokenized inputs before the model call.

        Override in the concrete class when the model's ``forward`` requires
        additional tensors that the processor does not produce (e.g. an
        explicit ``decoder_input_ids`` for seq2seq models).

        Args:
            model: The instantiated model.
            inputs: The ``BatchEncoding`` / dict returned by the processor.

        Returns:
            The (possibly modified) inputs dict.
        """
        return inputs

    # -----------------------------------------------------------------------
    # Tests
    # -----------------------------------------------------------------------

    @require_torch
    def test_fast_forward(self):
        """processor → model(**inputs): must not raise."""
        import torch

        self._skip_if_no_processor()
        raw_inputs = self._get_processor_inputs()

        for model_class in self.all_model_classes:
            with self.subTest(model_class=model_class.__name__):
                model = model_class.from_pretrained(self.model_id).eval()
                inputs = self.processor(**raw_inputs, return_tensors="pt", **self.processor_call_kwargs)
                inputs = self._prepare_model_inputs(model, inputs)
                with torch.no_grad():
                    outputs = model(**inputs)
                self.assertIsNotNone(outputs)

    @require_torch
    def test_fast_generate(self):
        """processor → model.generate(**inputs): must not raise."""
        import torch

        self._skip_if_no_processor()

        generative_classes = [c for c in self.all_model_classes if c.can_generate()]
        if not generative_classes:
            self.skipTest("No generative model classes in all_model_classes — skipping generate test")

        raw_inputs = self._get_processor_inputs()

        for model_class in generative_classes:
            with self.subTest(model_class=model_class.__name__):
                model = model_class.from_pretrained(self.model_id).eval()
                inputs = self.processor(**raw_inputs, return_tensors="pt", **self.processor_call_kwargs)
                inputs = self._prepare_model_inputs(model, inputs)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
                self.assertGreater(out.shape[-1], 0)
