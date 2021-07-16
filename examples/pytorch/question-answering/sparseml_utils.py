import inspect
import collections
import math
import os
from typing import Any

import numpy
import torch
import torch.nn.functional as F

import onnxruntime
from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.optim.optimizer import ScheduledOptimizer
from sparseml.pytorch.utils import ModuleExporter, logger
from trainer_qa import QuestionAnsweringTrainer
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.models.bert.modeling_bert import BertForQuestionAnswering


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


def export_model(model, dataloader, output_dir, num_exported_samples):
    """
    Export a trained model to ONNX
    :param model: trained model
    :param dataloader: dataloader to get sample batch
    :param output_dir: output directory for ONNX model
    """
    exporter = QuestionAnsweringModuleExporter(model, output_dir=output_dir)

    sess = None
    num_samples = 0

    sample_inputs = os.path.join(output_dir, "sample-inputs")
    sample_outputs = os.path.join(output_dir, "sample-outputs")
    os.makedirs(sample_inputs, exist_ok=True)
    os.makedirs(sample_outputs, exist_ok=True)

    forward_args_spec = inspect.getfullargspec(BertForQuestionAnswering.forward)
    for _, sample_batch in enumerate(dataloader):
        if sess is None:
            one_sample_input = collections.OrderedDict(
                [(f, sample_batch[f][0].reshape(1, -1)) for f in forward_args_spec.args if f in sample_batch]
            )

            try:
                exporter.export_onnx(sample_batch=one_sample_input, convert_qat=True)
                onnx_file = os.path.join(output_dir, "model.onnx")
            except Exception:
                raise RuntimeError("Error exporting ONNX models and/or inputs/outputs")

            sess = onnxruntime.InferenceSession(onnx_file)

        input_names = list(sample_batch.keys())
        output_names = [o.name for o in sess.get_outputs()]
        for input_vals in zip(*sample_batch.values()):
            input_feed = {k: v.reshape(1, -1).numpy() for k, v in zip(input_names, input_vals)}
            output_vals = sess.run(output_names, input_feed)
            output_dict = {name: val for name, val in zip(output_names, output_vals)}
            file_idx = f"{num_samples}".zfill(4)
            numpy.savez(f"{sample_inputs}/inp-{file_idx}.npz", **input_feed)
            numpy.savez(f"{sample_outputs}/out-{file_idx}.npz", **output_dict)
            num_samples += 1
            if num_samples >= num_exported_samples:
                return
