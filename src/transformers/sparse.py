import collections
import inspect
import math
import os
from typing import Optional

import numpy
import torch

import onnxruntime
from sparseml.pytorch.optim.manager import ScheduledModifierManager
from sparseml.pytorch.optim.optimizer import ScheduledOptimizer
from sparseml.pytorch.utils import logger
from transformers import Trainer
from transformers.file_utils import RECIPE_NAME, WEIGHTS_NAME


class SparseMLTrainer(Trainer):
    """
    Trainer with SparseML integration

    :param recipe: recipe for model sparsification
    :param teacher: teacher model for distillation
    :param distill_hardness: ratio of loss by teacher targets (between 0 and 1)
    :param distill_temperature: temperature for distillation
    :param args, kwargs: arguments passed into parent class
    """

    def __init__(
        self, model_name_or_path, recipes, teacher=None, distill_hardness=0.5, distill_temperature=2.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_name_or_path = str(model_name_or_path)
        self.recipes = [recipe for recipe in recipes if recipe]
        self.teacher = teacher
        self.distill_hardness = distill_hardness
        self.distill_temperature = distill_temperature
        self.criterion = torch.nn.CrossEntropyLoss()

        manager = None
        modifiers = []
        for recipe in self.recipes:
            manager = ScheduledModifierManager.from_yaml(recipe, modifiers)
            modifiers = manager.modifiers
        self.manager = manager

        self.loggers = None
        if self.recipes is not None:
            loggers = []
            if "wandb" in self.args.report_to:
                loggers.append(logger.WANDBLogger())
            self.loggers = loggers

    def apply_recipes(self, epoch=0.0):
        """
        Apply recipes and sparsification related parameters to the model
        """
        if self.manager is not None:
            org_state_dict = self.model.state_dict()
            self.manager.initialize(self.model, epoch=epoch, loggers=self.loggers)
            new_state_dict = self.model.state_dict()
            new_params = [p for p in new_state_dict.keys() if p not in org_state_dict]

            if os.path.isdir(self.model_name_or_path):
                if os.path.isfile(os.path.join(self.model_name_or_path, WEIGHTS_NAME)):
                    archive_file = os.path.join(self.model_name_or_path, WEIGHTS_NAME)
                    state_dict = torch.load(archive_file, map_location="cpu")
                    new_params_to_init = [p for p in new_params if p in state_dict.keys()]
                    if new_params_to_init:
                        # If we're here, the assumption is that all the new parameters introduced
                        # by the recipes are available to be restore from the checkpoint---this is
                        # case of evaluating pruned or pruned quantized models
                        # Otherwise, we're in use cases such as quantizing a block pruned model in which
                        # new parameters need to be initialized and trained during the QAT process
                        _, missing_keys, unexpected_keys, _ = self.model._load_state_dict_into_model(
                            self.model, state_dict, self.model_name_or_path, _fast_init=False
                        )
                        if missing_keys or unexpected_keys:
                            raise RuntimeError(
                                "Unexpected or missing keys detected when applying recipes to models\n"
                                f"Missing keys: {missing_keys}\n"
                                f"Unexpected keys: {unexpected_keys}\n"
                            )

    def create_optimizer(self):
        """
        Create optimizer customized using SparseML
        """
        super().create_optimizer()
        if not self.recipes:
            return
        steps_per_epoch = math.ceil(
            len(self.train_dataset) / (self.args.per_device_train_batch_size * self.args._n_gpu)
        )
        self.args.num_train_epochs = float(self.manager.max_epochs)
        if hasattr(self, "scaler"):
            self.scaler = self.manager.modify(
                self.model, self.optimizer, steps_per_epoch=steps_per_epoch, wrap_optim=self.scaler
            )
        else:
            self.optimizer = ScheduledOptimizer(
                self.optimizer, self.model, self.manager, steps_per_epoch=steps_per_epoch, loggers=self.loggers
            )

    def save_model(self, output_dir: Optional[str] = None):
        """
        Save model during or after training. The sparsification recipe will also be saved.
        """
        super().save_model(output_dir=output_dir)
        if self.manager is not None:
            self._save_recipe(output_dir=output_dir)

    def _save_recipe(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        output_recipe_file = os.path.join(output_dir, RECIPE_NAME)
        self.manager.save(output_recipe_file)


def export_model(exporter, dataloader, output_dir, num_exported_samples):
    """
    Export a trained model to ONNX
    :param exporter: a model exporter created from a trained model
    :param dataloader: dataloader to get sample batch
    :param output_dir: output directory for ONNX model
    """

    sess = None
    num_samples = 0

    sample_inputs = os.path.join(output_dir, "sample-inputs")
    sample_outputs = os.path.join(output_dir, "sample-outputs")
    os.makedirs(sample_inputs, exist_ok=True)
    os.makedirs(sample_outputs, exist_ok=True)

    for _, sample_batch in enumerate(dataloader):
        if sess is None:
            forward_args_spec = inspect.getfullargspec(exporter._module.__class__.forward)
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
            input_feed = {k: v.numpy() for k, v in zip(input_names, input_vals)}
            output_vals = sess.run(output_names, {k: input_feed[k].reshape(1, -1) for k in input_feed})
            output_dict = {name: numpy.squeeze(val) for name, val in zip(output_names, output_vals)}
            file_idx = f"{num_samples}".zfill(4)
            numpy.savez(f"{sample_inputs}/inp-{file_idx}.npz", **input_feed)
            numpy.savez(f"{sample_outputs}/out-{file_idx}.npz", **output_dict)
            num_samples += 1
            if num_samples >= num_exported_samples:
                return


def preprocess_state_dict(pretrained_model_name_or_path):
    """
    Restore original parameter names that were changed by QAT process
    """
    state_dict = None
    if pretrained_model_name_or_path is not None:
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, RECIPE_NAME)):
                recipe = os.path.join(pretrained_model_name_or_path, RECIPE_NAME)
                manager = ScheduledModifierManager.from_yaml(recipe)
                modifiers = [m.__class__.__name__ for m in manager.modifiers]
                is_qat_recipe = "QuantizationModifier" in modifiers
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                state_dict = torch.load(archive_file, map_location="cpu")
                removed_keys = (
                    [key for key in state_dict if (key.endswith(".module.weight") or key.endswith(".module.bias"))]
                    if is_qat_recipe
                    else []
                )
                for key in removed_keys:
                    new_key = key.replace(".module", "")
                    state_dict[new_key] = state_dict[key]
                    state_dict.pop(key)
    return state_dict


def load_recipe(pretrained_model_name_or_path):
    """
    Load recipe from the model directory
    """
    recipe = None
    if pretrained_model_name_or_path is not None:
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(os.path.join(pretrained_model_name_or_path, RECIPE_NAME)):
                recipe = os.path.join(pretrained_model_name_or_path, RECIPE_NAME)
    return recipe
