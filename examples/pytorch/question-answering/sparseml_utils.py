from typing import Any

import numpy
import torch
import torch.nn.functional as F

from sparseml.pytorch.utils import ModuleExporter
from trainer_qa import QuestionAnsweringTrainer
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.sparse import SparseMLTrainer


class SparseMLQATrainer(SparseMLTrainer, QuestionAnsweringTrainer):
    """
    Question Answering trainer with SparseML integration

    :param recipe: recipe for model sparsification
    :param teacher: teacher model for distillation
    :param distill_hardness: ratio of loss by teacher targets (between 0 and 1)
    :param distill_temperature: temperature for distillation
    :param args, kwargs: arguments passed into parent class
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computing loss using teacher/student distillation
        """
        if not self.recipes or self.teacher is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        outputs = model(**inputs)
        if self.teacher is None:
            loss = outputs["loss"]
        else:
            input_device = inputs["input_ids"].device
            self.teacher = self.teacher.to(input_device)
            start_logits_student = outputs["start_logits"]
            end_logits_student = outputs["end_logits"]
            start_logits_label = inputs["start_positions"]
            end_logits_label = inputs["end_positions"]
            with torch.no_grad():
                teacher_output = self.teacher(
                    input_ids=inputs["input_ids"],
                    token_type_ids=inputs["token_type_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            start_logits_teacher = teacher_output["start_logits"]
            end_logits_teacher = teacher_output["end_logits"]
            loss_start = (
                F.kl_div(
                    input=F.log_softmax(start_logits_student / self.distill_temperature, dim=-1),
                    target=F.softmax(start_logits_teacher / self.distill_temperature, dim=-1),
                    reduction="batchmean",
                )
                * (self.distill_temperature ** 2)
            )
            loss_end = (
                F.kl_div(
                    input=F.log_softmax(end_logits_student / self.distill_temperature, dim=-1),
                    target=F.softmax(end_logits_teacher / self.distill_temperature, dim=-1),
                    reduction="batchmean",
                )
                * (self.distill_temperature ** 2)
            )
            teacher_loss = (loss_start + loss_end) / 2.0
            loss_start = self.criterion(start_logits_student, start_logits_label)
            loss_end = self.criterion(end_logits_student, end_logits_label)
            label_loss = (loss_start + loss_end) / 2.0
            loss = ((1 - self.distill_hardness) * label_loss) + (self.distill_hardness * teacher_loss)
        return (loss, outputs) if return_outputs else loss


class QuestionAnsweringModuleExporter(ModuleExporter):
    """
    Module exporter class for Question Answering
    """

    @classmethod
    def get_output_names(self, out: Any):
        if not isinstance(out, QuestionAnsweringModelOutput):
            raise ValueError("Expected QuestionAnsweringModelOutput, got {type(out)}")
        expected = ["start_logits", "end_logits"]
        if numpy.any([name for name in expected if name not in out]):
            raise ValueError("Expected output names not found in model output")
        return expected
