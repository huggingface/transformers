from typing import List

from torch import nn

from transformers import BartConfig, BartForConditionalGeneration


def simple_adaptor(batch, model_outputs):
    # for textbrewer
    # The second and third elements of model outputs are the logits and hidden states
    return {"logits": model_outputs[0], "hidden": model_outputs[1]}


def make_teacher_and_student(teacher_cfg_kwargs, **student_updates):
    teacher_model = BartForConditionalGeneration(BartConfig(**teacher_cfg_kwargs))
    student_cfg_kwargs = teacher_cfg_kwargs.copy()
    student_cfg_kwargs.update(**student_updates)
    student_model = BartForConditionalGeneration(BartConfig(**student_cfg_kwargs))
    return teacher_model, student_model


def init_student(student, teacher):
    teacher_state_dict = teacher.state_dict()
    info = student.load_state_dict(teacher_state_dict, strict=False)
    assert info.missing_keys == [], info.missing_keys
    return student, info


def copy_decoder_layers(teacher, student, l2copy=[0, 2, 4, 7, 9, 11]):
    copy_layers(teacher.model.decoder.layers, student.model.decoder.layers, l2copy)


def copy_layers(teacher_layers, student_layers, l2copy: List):
    layers_to_copy = nn.ModuleList([l for i, l in enumerate(teacher_layers) if i in l2copy])
    assert len(student_layers) == len(l2copy), f"{len(student_layers)} != {len(l2copy)}"
    student_layers.load_state_dict(layers_to_copy.state_dict())
