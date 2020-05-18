from transformers import BartConfig, BartForConditionalGeneration


def simple_adaptor(batch, model_outputs):
    # The second and third elements of model outputs are the logits and hidden states
    return {"logits": model_outputs[0], "hidden": model_outputs[1]}


def make_teacher_and_student(teacher_cfg_kwargs, **student_updates):
    teacher_model = BartForConditionalGeneration(BartConfig(**teacher_cfg_kwargs))
    student_cfg_kwargs = teacher_cfg_kwargs.copy()
    student_cfg_kwargs.update(**student_updates)
    student_model = BartForConditionalGeneration(BartConfig(**student_cfg_kwargs))
    return teacher_model, student_model
