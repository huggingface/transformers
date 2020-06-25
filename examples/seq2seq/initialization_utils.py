from typing import List

from torch import nn


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
