import math

import torch
import torch.nn.functional as F

from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.optim.optimizer import ScheduledOptimizer
from sparseml.pytorch.utils import ModuleExporter, logger
from trainer_qa import QuestionAnsweringTrainer


class SparseMLQATrainer(QuestionAnsweringTrainer):
    """
    Question Answering trainer with SparseML integration

    :param recipe: recipe for model sparsification
    :param teacher: teacher model for distillation
    :param distill_hardness: ratio of loss by teacher targets (between 0 and 1)
    :param distill_temperature: temperature for distillation
    :param args, kwargs: arguments passed into parent class
    """

    def __init__(self, recipe, teacher=None, distill_hardness=0.5, distill_temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recipe = recipe
        self.teacher = teacher
        self.distill_hardness = distill_hardness
        self.distill_temperature = distill_temperature
        self.criterion = torch.nn.CrossEntropyLoss()

        self.manager = None
        self.loggers = None
        if self.recipe is not None:
            loggers = []
            if "wandb" in self.args.report_to:
                loggers.append(logger.WANDBLogger())
            self.loggers = loggers

    def create_optimizer(self):
        """
        Create optimizer customized using SparseML
        """
        super().create_optimizer()
        if self.recipe is None:
            return
        steps_per_epoch = math.ceil(
            len(self.train_dataset) / (self.args.per_device_train_batch_size * self.args._n_gpu)
        )
        self.manager = ScheduledModifierManager.from_yaml(self.recipe)
        self.args.num_train_epochs = float(self.manager.max_epochs)
        if hasattr(self, "scaler"):
            self.manager.initialize(self.model, epoch=0.0, loggers=self.loggers)
            self.scaler = self.manager.modify(
                self.model, self.optimizer, steps_per_epoch=steps_per_epoch, wrap_optim=self.scaler
            )
        else:
            self.optimizer = ScheduledOptimizer(
                self.optimizer, self.model, self.manager, steps_per_epoch=steps_per_epoch, loggers=self.loggers
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computing loss using teacher/student distillation
        """
        if self.recipe is None or self.teacher is None:
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


def export_model(model, dataloader, output_dir):
    """
    Export a trained model to ONNX
    :param model: trained model
    :param dataloader: dataloader to get sample batch
    :param output_dir: output directory for ONNX model
    """
    exporter = ModuleExporter(model, output_dir=output_dir)
    for _, sample_batch in enumerate(dataloader):
        sample_input = (sample_batch["input_ids"], sample_batch["attention_mask"], sample_batch["token_type_ids"])
        exporter.export_onnx(sample_batch=sample_input, convert_qat=True)
        break
