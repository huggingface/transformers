import warnings
from pathlib import Path
from typing import List, Tuple, Union

import fire
from torch import nn

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel
from transformers.utils import logging


logger = logging.get_logger(__name__)


def copy_layers(src_layers: nn.ModuleList, dest_layers: nn.ModuleList, layers_to_copy: List[int]) -> None:
    layers_to_copy = nn.ModuleList([src_layers[i] for i in layers_to_copy])
    assert len(dest_layers) == len(layers_to_copy), f"{len(dest_layers)} != {len(layers_to_copy)}"
    dest_layers.load_state_dict(layers_to_copy.state_dict())


LAYERS_TO_COPY = {
    # maps  num layers in teacher -> num_layers in student -> which teacher layers to copy.
    # 12: bart, 16: pegasus, 6: marian/Helsinki-NLP
    12: {
        1: [0],  # This says that if the teacher has 12 layers and the student has 1, copy layer 0 of the teacher
        2: [0, 6],
        3: [0, 6, 11],
        4: [0, 4, 8, 11],
        6: [0, 2, 4, 7, 9, 11],
        9: [0, 1, 2, 4, 5, 7, 9, 10, 11],
        12: list(range(12)),
    },
    16: {  # maps  num layers in student -> which teacher layers to copy
        1: [0],
        2: [0, 15],
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
    # maps  num layers in student -> which teacher layers to copy.
    6: {1: [5], 2: [3, 5], 3: [1, 4, 5], 4: [1, 2, 4, 5]},
    12: {1: [11], 2: [5, 11], 3: [3, 7, 11], 6: [1, 3, 5, 8, 10, 11]},
    16: {1: [15], 4: [4, 9, 12, 15], 8: [1, 3, 5, 7, 9, 11, 13, 15]},
}


def pick_layers_to_copy(n_student, n_teacher):
    try:
        val = LAYERS_TO_COPY[n_teacher][n_student]
        return val
    except KeyError:
        if n_student != n_teacher:
            warnings.warn(
                f"no hardcoded layers to copy for teacher {n_teacher} -> student {n_student}, defaulting to first {n_student}"
            )
        return list(range(n_student))


def get_layers_to_supervise(n_student, n_teacher) -> List[int]:
    """Used or the --supervise_forward kwarg"""
    if n_student > n_teacher:
        raise ValueError(f"Cannot perform intermediate supervision for student {n_student} > teacher {n_teacher}")
    elif n_teacher == n_student:
        return list(range(n_teacher))
    elif n_student == 1:
        return [n_teacher - 1]
    else:
        return LAYERS_TO_SUPERVISE[n_teacher][n_student]


def create_student_by_copying_alternating_layers(
    teacher: Union[str, PreTrainedModel],
    save_path: Union[str, Path] = "student",
    e: Union[int, None] = None,
    d: Union[int, None] = None,
    copy_first_teacher_layers=False,
    e_layers_to_copy=None,
    d_layers_to_copy=None,
    **extra_config_kwargs
) -> Tuple[PreTrainedModel, List[int], List[int]]:
    """Make a student by copying alternating layers from a teacher, save it to save_path.
    Args:
        teacher: str or PreTrainedModel if str, this will call AutoModelForSeq2SeqLM.from_pretrained(teacher) before
        copying layers
        save_path: where to save the student, defaults to student directory.
        e: how many Encoder layers should the student have, default is fully copy of teacher
        d: how many Decoder layers should the student have, default is fully copy of teacher
        copy_first_teacher_layers: [bool] dont copy alternating layers, just the first e/d.
        **extra_config_kwargs: extra kwargs to pass to the student, by default the teacher config is used.

    Returns:
        student: new, smaller model.  (Also saves it to save_path)
        e_layers_to_copy: list of which teacher encoder layers were used
        d_layers_to_copy: list of which teacher decoder layers were used
    """
    _msg = "encoder_layers and decoder_layers cannot be both None-- you would just have an identical teacher."
    assert (e is not None) or (d is not None), _msg
    if isinstance(teacher, str):
        AutoTokenizer.from_pretrained(teacher).save_pretrained(save_path)  # purely for convenience
        teacher = AutoModelForSeq2SeqLM.from_pretrained(teacher).eval()
    else:

        assert isinstance(teacher, PreTrainedModel), f"teacher must be a model or string got type {type(teacher)}"
    init_kwargs = teacher.config.to_diff_dict()

    try:
        teacher_e, teacher_d = teacher.config.encoder_layers, teacher.config.decoder_layers
        if e is None:
            e = teacher_e
        if d is None:
            d = teacher_d
        init_kwargs.update({"encoder_layers": e, "decoder_layers": d})
    except AttributeError:  # T5
        teacher_e, teacher_d = teacher.config.num_layers, teacher.config.num_decoder_layers
        if e is None:
            e = teacher_e
        if d is None:
            d = teacher_d
        init_kwargs.update({"num_layers": e, "num_decoder_layers": d})

    # Kwargs to instantiate student: teacher kwargs with updated layer numbers + **extra_config_kwargs
    init_kwargs.update(extra_config_kwargs)

    # Copy weights
    student_cfg = teacher.config_class(**init_kwargs)
    student = AutoModelForSeq2SeqLM.from_config(student_cfg)
    # Start by copying the full teacher state dict this will copy the first N teacher layers to the student.
    info = student.load_state_dict(teacher.state_dict(), strict=False)
    assert info.missing_keys == [], info.missing_keys  # every student key should have a teacher keys.

    if copy_first_teacher_layers:  # Our copying is done. We just log and save
        e_layers_to_copy, d_layers_to_copy = list(range(e)), list(range(d))
        logger.info(
            f"Copied encoder layers {e_layers_to_copy} and decoder layers {d_layers_to_copy}. Saving them to {save_path}"
        )
        student.save_pretrained(save_path)
        return student, e_layers_to_copy, d_layers_to_copy

    # Decide which layers of the teacher to copy. Not exactly alternating -- we try to keep first and last layer.
    if e_layers_to_copy is None:
        e_layers_to_copy: List[int] = pick_layers_to_copy(e, teacher_e)
    if d_layers_to_copy is None:
        d_layers_to_copy: List[int] = pick_layers_to_copy(d, teacher_d)

    try:
        copy_layers(teacher.model.encoder.layers, student.model.encoder.layers, e_layers_to_copy)
        copy_layers(teacher.model.decoder.layers, student.model.decoder.layers, d_layers_to_copy)
    except AttributeError:  # For t5, student.model.encoder.layers is called student.encoder.block
        copy_layers(teacher.encoder.block, student.encoder.block, e_layers_to_copy)
        copy_layers(teacher.decoder.block, student.decoder.block, d_layers_to_copy)
    logger.info(
        f"Copied encoder layers {e_layers_to_copy} and decoder layers {d_layers_to_copy}. Saving them to {save_path}"
    )
    student.config.init_metadata = dict(
        teacher_type=teacher.config.model_type,
        copied_encoder_layers=e_layers_to_copy,
        copied_decoder_layers=d_layers_to_copy,
    )
    student.save_pretrained(save_path)
    # Save information about copying for easier reproducibility

    return student, e_layers_to_copy, d_layers_to_copy


if __name__ == "__main__":
    fire.Fire(create_student_by_copying_alternating_layers)
