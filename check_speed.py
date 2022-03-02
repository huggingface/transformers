import json
import os
import sys
import numpy as np
import tempfile
import tensorflow as tf
import torch
import time

sys.path.append("./")

from tests.bert.test_modeling_tf_bert import TFBertModelTest


torch_device = "cpu"


class Tester:

    def __init__(self, test_class):

        class _test_class(test_class):

            def test_pt_tf_model_equivalence_original(self, buf):
                import torch

                import transformers

                config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

                for model_class in self.all_model_classes:

                    buf[model_class.__name__] = []

                    for idx in range(100):

                        s = time.time()

                        pt_model_class_name = model_class.__name__[2:]  # Skip the "TF" at the beginning
                        pt_model_class = getattr(transformers, pt_model_class_name)

                        config.output_hidden_states = True

                        tf_model = model_class(config)
                        pt_model = pt_model_class(config)

                        # Check we can load pt model in tf and vice-versa with model => model functions
                        tf_model = transformers.load_pytorch_model_in_tf2_model(
                            tf_model, pt_model, tf_inputs=self._prepare_for_class(inputs_dict, model_class)
                        )
                        pt_model = transformers.load_tf2_model_in_pytorch_model(pt_model, tf_model)

                        # Check predictions on first output (logits/hidden-states) are close enought given low-level computational differences
                        pt_model.eval()
                        pt_inputs_dict = {}
                        for name, key in self._prepare_for_class(inputs_dict, model_class).items():
                            if type(key) == bool:
                                pt_inputs_dict[name] = key
                            elif name == "input_values":
                                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
                            elif name == "pixel_values":
                                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
                            elif name == "input_features":
                                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
                            else:
                                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.long)

                        with torch.no_grad():
                            pto = pt_model(**pt_inputs_dict)
                        tfo = tf_model(self._prepare_for_class(inputs_dict, model_class), training=False)

                        tf_hidden_states = tfo[0].numpy()
                        pt_hidden_states = pto[0].numpy()

                        tf_nans = np.copy(np.isnan(tf_hidden_states))
                        pt_nans = np.copy(np.isnan(pt_hidden_states))

                        pt_hidden_states[tf_nans] = 0
                        tf_hidden_states[tf_nans] = 0
                        pt_hidden_states[pt_nans] = 0
                        tf_hidden_states[pt_nans] = 0

                        max_diff = np.amax(np.abs(tf_hidden_states - pt_hidden_states))
                        self.assertLessEqual(max_diff, 4e-2)

                        # Check we can load pt model in tf and vice-versa with checkpoint => model functions
                        with tempfile.TemporaryDirectory() as tmpdirname:
                            pt_checkpoint_path = os.path.join(tmpdirname, "pt_model.bin")
                            torch.save(pt_model.state_dict(), pt_checkpoint_path)
                            tf_model = transformers.load_pytorch_checkpoint_in_tf2_model(tf_model, pt_checkpoint_path)

                            tf_checkpoint_path = os.path.join(tmpdirname, "tf_model.h5")
                            tf_model.save_weights(tf_checkpoint_path)
                            pt_model = transformers.load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path)

                        # Check predictions on first output (logits/hidden-states) are close enought given low-level computational differences
                        pt_model.eval()
                        pt_inputs_dict = {}
                        for name, key in self._prepare_for_class(inputs_dict, model_class).items():
                            if type(key) == bool:
                                key = np.array(key, dtype=bool)
                                pt_inputs_dict[name] = torch.from_numpy(key).to(torch.long)
                            elif name == "input_values":
                                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
                            elif name == "pixel_values":
                                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
                            elif name == "input_features":
                                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
                            else:
                                pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.long)

                        with torch.no_grad():
                            pto = pt_model(**pt_inputs_dict)
                        tfo = tf_model(self._prepare_for_class(inputs_dict, model_class))
                        tfo = tfo[0].numpy()
                        pto = pto[0].numpy()
                        tf_nans = np.copy(np.isnan(tfo))
                        pt_nans = np.copy(np.isnan(pto))

                        pto[tf_nans] = 0
                        tfo[tf_nans] = 0
                        pto[pt_nans] = 0
                        tfo[pt_nans] = 0

                        max_diff = np.amax(np.abs(tfo - pto))
                        self.assertLessEqual(max_diff, 4e-2)

                        e = time.time()

                        print(f"{model_class.__name__} - Elapsed time (previous test): {e-s}")

                        buf[model_class.__name__].append(float(e-s))

            def test_pt_tf_model_equivalence_new(self, buf):
                import torch

                import transformers

                def prepare_pt_inputs_from_tf_inputs(tf_inputs_dict):

                    pt_inputs_dict = {}
                    for name, key in tf_inputs_dict.items():
                        if type(key) == bool:
                            pt_inputs_dict[name] = key
                        elif name == "input_values":
                            pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
                        elif name == "pixel_values":
                            pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
                        elif name == "input_features":
                            pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
                        else:
                            pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.long)

                    return pt_inputs_dict

                def check_outputs(tfo, pto, model_class, names):
                    """
                    Args:
                        model_class: The class of the model that is currently testing. For example, `TFBertModel`,
                            TFBertForMaskedLM`, `TFBertForSequenceClassification`, etc. Currently unused, but it could make
                            debugging easier and faster.

                        names: A string, or a list of strings. These specify what tfo/pto represent in the model outputs.
                            Currently unused, but in the future, we could use this information to make the error message clearer
                            by giving the name(s) of the output tensor(s) with large difference(s) between PT and TF.
                    """

                    # Some issue (`about past_key_values`) to solve (e.g. `TFPegasusForConditionalGeneration`) in a separate PR.
                    if names == "past_key_values":
                        return

                    if type(tfo) == tuple:
                        self.assertEqual(type(pto), tuple)
                        self.assertEqual(len(tfo), len(pto))
                        if type(names) in [tuple, list]:
                            for to, po, name in zip(tfo, pto, names):
                                check_outputs(to, po, model_class, names=name)
                        elif type(names) == str:
                            for idx, (to, po) in enumerate(zip(tfo, pto)):
                                check_outputs(to, po, model_class, names=f"{names}_{idx}")
                    elif isinstance(tfo, tf.Tensor):
                        self.assertTrue(isinstance(pto, torch.Tensor))

                        tfo = tfo.numpy()
                        pto = pto.detach().to("cpu").numpy()

                        tf_nans = np.copy(np.isnan(tfo))
                        pt_nans = np.copy(np.isnan(pto))

                        pto[tf_nans] = 0
                        tfo[tf_nans] = 0
                        pto[pt_nans] = 0
                        tfo[pt_nans] = 0

                        max_diff = np.amax(np.abs(tfo - pto))
                        self.assertLessEqual(max_diff, 1e-5)

                for model_class in self.all_model_classes:

                    buf[model_class.__name__] = []

                    for idx in range(100):

                        s = time.time()

                        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

                        # Output all for aggressive testing
                        config.output_hidden_states = True
                        # Pure convolutional models have no attention
                        # TODO: use a better and general criteria
                        if "TFConvNext" not in model_class.__name__:
                            config.output_attentions = True

                        # TODO: remove this block once the large negative value for attention masks is fixed.
                        for k in ["attention_mask", "encoder_attention_mask", "decoder_attention_mask"]:
                            if k in inputs_dict:
                                attention_mask = inputs_dict[k]
                                # (make sure no all 0s attention masks - to avoid failure at this moment)
                                attention_mask = tf.ones_like(attention_mask, dtype=tf.int32)
                                # (make the first sequence with all 0s attention mask -> to demonstrate the issue)
                                # (this will fail for `TFWav2Vec2Model`)
                                # attention_mask = tf.concat(
                                #     [
                                #         tf.zeros_like(attention_mask[:1], dtype=tf.int32),
                                #         tf.cast(attention_mask[1:], dtype=tf.int32)
                                #     ],
                                #     axis=0
                                # )
                                inputs_dict[k] = attention_mask

                        pt_model_class_name = model_class.__name__[2:]  # Skip the "TF" at the beginning
                        pt_model_class = getattr(transformers, pt_model_class_name)

                        config.output_hidden_states = True

                        tf_model = model_class(config)
                        pt_model = pt_model_class(config)

                        tf_inputs_dict = self._prepare_for_class(inputs_dict, model_class)
                        tf_inputs_dict_maybe_with_labels = self._prepare_for_class(inputs_dict, model_class, return_labels=True)

                        # Check we can load pt model in tf and vice-versa with model => model functions
                        tf_model = transformers.load_pytorch_model_in_tf2_model(tf_model, pt_model, tf_inputs=tf_inputs_dict)
                        pt_model = transformers.load_tf2_model_in_pytorch_model(pt_model, tf_model)

                        # send pytorch model to the correct device
                        pt_model.to(torch_device)

                        # Check predictions on first output (logits/hidden-states) are close enough given low-level computational differences
                        pt_model.eval()

                        pt_inputs_dict = prepare_pt_inputs_from_tf_inputs(tf_inputs_dict)
                        pt_inputs_dict_maybe_with_labels = prepare_pt_inputs_from_tf_inputs(tf_inputs_dict_maybe_with_labels)

                        # send pytorch inputs to the correct device
                        pt_inputs_dict = {
                            k: v.to(device=torch_device) if isinstance(v, torch.Tensor) else v for k, v in
                            pt_inputs_dict.items()
                        }
                        pt_inputs_dict_maybe_with_labels = {
                            k: v.to(device=torch_device) if isinstance(v, torch.Tensor) else v
                            for k, v in pt_inputs_dict_maybe_with_labels.items()
                        }

                        # Original test: check without `labels`
                        with torch.no_grad():
                            pto = pt_model(**pt_inputs_dict)
                        tfo = tf_model(tf_inputs_dict)

                        tf_keys = [k for k, v in tfo.items() if v is not None]
                        pt_keys = [k for k, v in pto.items() if v is not None]

                        self.assertEqual(tf_keys, pt_keys)
                        check_outputs(tfo, pto, model_class, names=tf_keys)

                        # check the case where `labels` is passed
                        has_labels = any(
                            x in tf_inputs_dict_maybe_with_labels for x in ["labels", "next_sentence_label", "start_positions"]
                        )
                        if has_labels:

                            with torch.no_grad():
                                pto = pt_model(**pt_inputs_dict_maybe_with_labels)
                            tfo = tf_model(tf_inputs_dict_maybe_with_labels)

                            # Some models' output class don't have `loss` attribute despite `labels` is used.
                            # TODO: identify which models
                            tf_loss = getattr(tfo, "loss", None)
                            pt_loss = getattr(pto, "loss", None)

                            # Some PT models return loss while the corresponding TF models don't (i.e. `None` for `loss`).
                            #   - TFFlaubertWithLMHeadModel
                            #   - TFFunnelForPreTraining
                            #   - TFElectraForPreTraining
                            #   - TFXLMWithLMHeadModel
                            # TODO: Fix PT/TF diff -> remove this condition to fail the test if a diff occurs
                            if not ((tf_loss is None and pt_loss is None) or (tf_loss is not None and pt_loss is not None)):
                                if model_class.__name__ not in [
                                    "TFFlaubertWithLMHeadModel",
                                    "TFFunnelForPreTraining",
                                    "TFElectraForPreTraining",
                                    "TFXLMWithLMHeadModel",
                                ]:
                                    self.assertEqual(tf_loss is None, pt_loss is None)

                            tf_keys = [k for k, v in tfo.items() if v is not None]
                            pt_keys = [k for k, v in pto.items() if v is not None]

                            # TODO: remove these 2 conditions once the above TODOs (above loss) are implemented
                            # (Also, `TFTransfoXLLMHeadModel` has no `loss` while `TransfoXLLMHeadModel` return `losses`)
                            if tf_keys != pt_keys:
                                if model_class.__name__ not in [
                                    "TFFlaubertWithLMHeadModel",
                                    "TFFunnelForPreTraining",
                                    "TFElectraForPreTraining",
                                    "TFXLMWithLMHeadModel",
                                ] + ["TFTransfoXLLMHeadModel"]:
                                    self.assertEqual(tf_keys, pt_keys)

                            # Since we deliberately make some tests pass above (regarding the `loss`), let's still try to test
                            # some remaining attributes in the outputs.
                            # TODO: remove this block of `index` computing once the above TODOs (above loss) are implemented
                            # compute the 1st `index` where `tf_keys` and `pt_keys` is different
                            index = 0
                            for _ in range(min(len(tf_keys), len(pt_keys))):
                                if tf_keys[index] == pt_keys[index]:
                                    index += 1
                                else:
                                    break
                            if tf_keys[:index] != pt_keys[:index]:
                                self.assertEqual(tf_keys, pt_keys)

                            # Some models require extra condition to return loss. For example, `(TF)BertForPreTraining` requires
                            # both`labels` and `next_sentence_label`.
                            if tf_loss is not None and pt_loss is not None:

                                # check anything else than `loss`
                                keys = [k for k in tf_keys]
                                check_outputs(tfo[1:index], pto[1:index], model_class, names=keys[1:index])

                                # check `loss`

                                # tf models returned loss is usually a tensor rather than a scalar.
                                # (see `hf_compute_loss`: it uses `tf.keras.losses.Reduction.NONE`)
                                # Change it here to a scalar to match PyTorch models' loss
                                tf_loss = tf.math.reduce_mean(tf_loss).numpy()
                                pt_loss = pt_loss.detach().to("cpu").numpy()

                                tf_nans = np.copy(np.isnan(tf_loss))
                                pt_nans = np.copy(np.isnan(pt_loss))
                                # the 2 losses need to be both nan or both not nan
                                self.assertEqual(tf_nans, pt_nans)

                                if not tf_nans:
                                    max_diff = np.amax(np.abs(tf_loss - pt_loss))
                                    self.assertLessEqual(max_diff, 1e-5)

                        e = time.time()

                        print(f"{model_class.__name__} - Elapsed time (new test): {e - s}")

                        buf[model_class.__name__].append(float(e - s))

        test = _test_class()
        test.setUp()

        print(test.model_tester)
        print(test.config_tester)

        self.test = test

tester = Tester(TFBertModelTest)

s1 = time.time()

results = {
    "original": {},
    "new": {}
}

tester.test.test_pt_tf_model_equivalence_original(results["original"])
tester.test.test_pt_tf_model_equivalence_new(results["new"])


r = {}
for k in results["original"]:
    r[k] = {
        "original": results["original"][k],
        "new": results["new"][k],
    }

s = json.dumps(r, indent=4, ensure_ascii=False)
print(s)

with open("test_timing.json", "w", encoding="UTF-8") as fp:
    json.dump(r, fp, indent=4, ensure_ascii=False)
