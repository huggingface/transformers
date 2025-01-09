import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(ROOT_DIR, "utils"))

import create_dependency_mapping  # noqa: E402

# This is equivalent to `all` in the current library state (as of 09/01/2025)
model_root = os.path.join("src", "transformers", "models")
files_to_parse = [
    os.path.join(model_root, "starcoder2", "modular_starcoder2.py"),
    os.path.join(model_root, "gemma", "modular_gemma.py"),
    os.path.join(model_root, "olmo2", "modular_olmo2.py"),
    os.path.join(model_root, "diffllama", "modular_diffllama.py"),
    os.path.join(model_root, "granite", "modular_granite.py"),
    os.path.join(model_root, "gemma2", "modular_gemma2.py"),
    os.path.join(model_root, "mixtral", "modular_mixtral.py"),
    os.path.join(model_root, "olmo", "modular_olmo.py"),
    os.path.join(model_root, "rt_detr", "modular_rt_detr.py"),
    os.path.join(model_root, "qwen2", "modular_qwen2.py"),
    os.path.join(model_root, "llava_next_video", "modular_llava_next_video.py"),
    os.path.join(model_root, "cohere2", "modular_cohere2.py"),
    os.path.join(model_root, "modernbert", "modular_modernbert.py"),
    os.path.join(model_root, "colpali", "modular_colpali.py"),
    os.path.join(model_root, "deformable_detr", "modular_deformable_detr.py"),
    os.path.join(model_root, "aria", "modular_aria.py"),
    os.path.join(model_root, "ijepa", "modular_ijepa.py"),
    os.path.join(model_root, "bamba", "modular_bamba.py"),
    os.path.join(model_root, "dinov2_with_registers", "modular_dinov2_with_registers.py"),
    os.path.join(model_root, "instructblipvideo", "modular_instructblipvideo.py"),
    os.path.join(model_root, "glm", "modular_glm.py"),
    os.path.join(model_root, "phi", "modular_phi.py"),
    os.path.join(model_root, "mistral", "modular_mistral.py"),
]

# Find the order
priority_list = create_dependency_mapping.find_priority_list(files_to_parse)
# Extract just the model names
model_priority_list = []

def appear_after(model1: str, file2: str) -> bool:
    pass
