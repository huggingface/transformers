import argparse
import warnings
from typing import List

from torch import nn


try:
    from .distillation import LAYERS_TO_COPY
except ImportError:
    from distillation import LAYERS_TO_COPY

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def get_layers_to_copy(n_student, n_teacher):
    try:
        val = LAYERS_TO_COPY[n_teacher][n_student]
        return val
    except KeyError:
        if n_student != n_teacher:
            warnings.warn(
                f"no hardcoded layers to copy for teacher {n_teacher} -> student {n_student}, defaulting to first {n_student}"
            )
        return list(range(n_student))


def copy_to_student(
    d_layers_to_copy, e_layers_to_copy, student_encoder_layers, student_decoder_layers, student, teacher
):
    different_encoder: bool = student_encoder_layers != teacher.config.encoder_layers
    different_decoder = student_decoder_layers != teacher.config.decoder_layers
    if different_decoder:
        copy_layers(teacher.model.decoder.layers, student.model.decoder.layers, d_layers_to_copy)
    if different_encoder:
        copy_layers(teacher.model.encoder.layers, student.model.encoder.layers, e_layers_to_copy)


def init_student(student, teacher):
    teacher_state_dict = teacher.state_dict()
    info = student.load_state_dict(teacher_state_dict, strict=False)
    assert info.missing_keys == [], info.missing_keys
    return student, info


def copy_decoder_layers(teacher, student, l2copy=[0, 2, 4, 7, 9, 11]):
    copy_layers(teacher.model.decoder.layers, student.model.decoder.layers, l2copy)


def copy_layers(teacher_layers: nn.ModuleList, student_layers: nn.ModuleList, layers_to_copy: List) -> None:
    layers_to_copy = nn.ModuleList([l for i, l in enumerate(teacher_layers) if i in layers_to_copy])
    assert len(student_layers) == len(layers_to_copy), f"{len(student_layers)} != {len(layers_to_copy)}"
    student_layers.load_state_dict(layers_to_copy.state_dict())


def create_student(teacher, student_encoder_layers, student_decoder_layers):
    assert (
        student_encoder_layers is not None or student_decoder_layers is not None
    ), "student_encoder_layers and student_decoder_layers both cannot be None, please specify at least one of them."

    teacher = AutoModelForSeq2SeqLM.from_pretrained(teacher).eval()
    student_encoder_layers = (
        student_encoder_layers if student_encoder_layers is not None else teacher.config.encoder_layers
    )
    student_decoder_layers = (
        student_decoder_layers if student_decoder_layers is not None else teacher.config.decoder_layers
    )
    student_updates = {
        "decoder_layers": student_decoder_layers,
        "encoder_layers": student_encoder_layers,
    }
    e_layers_to_copy: List = get_layers_to_copy(student_updates["encoder_layers"], teacher.config.encoder_layers)
    d_layers_to_copy: List = get_layers_to_copy(student_updates["decoder_layers"], teacher.config.decoder_layers)

    kw = teacher.config.to_diff_dict()
    kw.update(student_updates)
    # Copy weights
    student_cfg = teacher.config_class(**kw)
    student = type(teacher)(student_cfg)
    student, _ = init_student(student, teacher)
    copy_to_student(
        d_layers_to_copy, e_layers_to_copy, student_encoder_layers, student_decoder_layers, student, teacher
    )
    return student


def main():
    parser = argparse.ArgumentParser(description="Creat student model for distilation.")
    parser.add_argument("--teacher_model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--student_encoder_layers", type=int, required=True, default=None, help="# encoder layers for student"
    )
    parser.add_argument(
        "--student_decoder_layers", type=int, required=True, default=None, help="# decoder layers for student"
    )
    parser.add_argument("--save_path", type=str, required=True, help="Where to save student model")

    args = parser.parse_args()

    student = create_student(args.teacher_model_name_or_path, args.student_encoder_layers, args.student_decoder_layers)
    student.save_pretrained(args.save_path)
    # save tokenizer as well
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name_or_path)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main()
