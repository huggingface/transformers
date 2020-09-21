import argparse
import os
import warnings
from typing import List

from torch import nn

from transformers import AutoModelForSeq2SeqLM


LAYERS_TO_COPY = {
    # maps  num layers in student -> which teacher layers to copy.
    # 12: bart, 16: pegasus, 6: marian/Helsinki-NLP
    12: {
        1: [0],
        2: [0, 6],
        3: [0, 6, 11],
        4: [0, 4, 8, 11],
        6: [0, 2, 4, 7, 9, 11],
        9: [0, 1, 2, 4, 5, 7, 9, 10, 11],
        12: list(range(12)),
    },
    16: {  # maps  num layers in student -> which teacher layers to copy
        1: [0],
        2: [0, 8],
        3: [0, 8, 15],
        4: [0, 5, 10, 15],
        6: [0, 3, 6, 9, 12, 15],
        8: [0, 2, 4, 6, 8, 10, 12, 15],
        9: [0, 1, 3, 5, 7, 9, 11, 13, 15],
        12: [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15],
        16: list(range(16)),
    },
    6: {1: [0], 2: [0, 5], 3: [0, 2, 5], 4: [0, 1, 3, 5], 6: list(range(6))},
}

LAYERS_TO_SUPERVISE = {
    12: {1: [11], 2: [5, 11], 3: [3, 7, 11], 6: [1, 3, 5, 8, 10, 11]},
    16: {1: [15], 4: [4, 9, 12, 15], 8: [1, 3, 5, 7, 9, 11, 13, 15]},
    6: {1: [5], 2: [3, 5], 3: [1, 4, 5], 4: [1, 2, 4, 5]},
    2: {1: [1], 2: [0, 1]},
}


def get_layers_to_copy(n_student, n_teacher):
    try:
        val = LAYERS_TO_COPY[n_teacher][n_student]
        # assert len(LAYERS_TO_SUPERVISE[n_teacher][n_student]) == len(val) == n_student
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
    teacher = AutoModelForSeq2SeqLM.from_pretrained(teacher).eval()
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
    parser = argparse.ArgumentParser(description="Creat BART student model for sequence classification.")
    parser.add_argument("--teacher_model_name_or_path", type=str, required=True)
    parser.add_argument("--student_encoder_layers", type=int, required=True, help="# encoder layers for student")
    parser.add_argument("--student_decoder_layers", type=int, required=True, help="# decoder layers for student")
    parser.add_argument("--save_path", type=str, required=True, help="Where to save student model")

    args = parser.parse_args()

    student = create_student(args.teacher_model_name_or_path, args.student_encoder_layers, args.student_decoder_layers)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    student.save_pretrained(args.save_path)


if __name__ == "__main__":
    main()
