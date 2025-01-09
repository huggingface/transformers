import os
import sys
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(ROOT_DIR, "utils"))

import create_dependency_mapping  # noqa: E402

# This is equivalent to `all` in the current library state (as of 09/01/2025)
MODEL_ROOT = os.path.join("src", "transformers", "models")
FILES_TO_PARSE = [
    os.path.join(MODEL_ROOT, "starcoder2", "modular_starcoder2.py"),
    os.path.join(MODEL_ROOT, "gemma", "modular_gemma.py"),
    os.path.join(MODEL_ROOT, "olmo2", "modular_olmo2.py"),
    os.path.join(MODEL_ROOT, "diffllama", "modular_diffllama.py"),
    os.path.join(MODEL_ROOT, "granite", "modular_granite.py"),
    os.path.join(MODEL_ROOT, "gemma2", "modular_gemma2.py"),
    os.path.join(MODEL_ROOT, "mixtral", "modular_mixtral.py"),
    os.path.join(MODEL_ROOT, "olmo", "modular_olmo.py"),
    os.path.join(MODEL_ROOT, "rt_detr", "modular_rt_detr.py"),
    os.path.join(MODEL_ROOT, "qwen2", "modular_qwen2.py"),
    os.path.join(MODEL_ROOT, "llava_next_video", "modular_llava_next_video.py"),
    os.path.join(MODEL_ROOT, "cohere2", "modular_cohere2.py"),
    os.path.join(MODEL_ROOT, "modernbert", "modular_modernbert.py"),
    os.path.join(MODEL_ROOT, "colpali", "modular_colpali.py"),
    os.path.join(MODEL_ROOT, "deformable_detr", "modular_deformable_detr.py"),
    os.path.join(MODEL_ROOT, "aria", "modular_aria.py"),
    os.path.join(MODEL_ROOT, "ijepa", "modular_ijepa.py"),
    os.path.join(MODEL_ROOT, "bamba", "modular_bamba.py"),
    os.path.join(MODEL_ROOT, "dinov2_with_registers", "modular_dinov2_with_registers.py"),
    os.path.join(MODEL_ROOT, "instructblipvideo", "modular_instructblipvideo.py"),
    os.path.join(MODEL_ROOT, "glm", "modular_glm.py"),
    os.path.join(MODEL_ROOT, "phi", "modular_phi.py"),
    os.path.join(MODEL_ROOT, "mistral", "modular_mistral.py"),
]


def appear_after(model1: str, model2: str, priority_list: list[str]) -> bool:
    """Return True if `model1` appear after `model2` in `priority_list`."""
    return priority_list.index(model1) > priority_list.index(model2)


class ConversionOrderTest(unittest.TestCase):

    def test_conversion_order(self):
        # Find the order
        priority_list = create_dependency_mapping.find_priority_list(FILES_TO_PARSE)
        # Extract just the model names
        model_priority_list = [file.rsplit("modular_")[-1].replace(".py", "") for file in priority_list]

        # These are based on what the current library order should be (as of 09/01/2025)
        self.assertTrue(appear_after("mixtral", "mistral", model_priority_list))
        self.assertTrue(appear_after("gemma2", "gemma", model_priority_list))
        self.assertTrue(appear_after("starcoder2", "mistral", model_priority_list))
        self.assertTrue(appear_after("olmo2", "olmo", model_priority_list))
        self.assertTrue(appear_after("diffllama", "mistral", model_priority_list))
        self.assertTrue(appear_after("cohere2", "gemma2", model_priority_list))