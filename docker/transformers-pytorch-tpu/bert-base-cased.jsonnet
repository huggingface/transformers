local base = import 'templates/base.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import "templates/utils.libsonnet";
local volumes = import "templates/volumes.libsonnet";

local bertBaseCased = base.BaseTest {
  frameworkPrefix: "hf",
  modelName: "bert-base-cased",
  mode: "example",
  configMaps: [],

  timeout: 3600, # 1 hour, in seconds

  image: std.extVar('image'),
  imageTag: std.extVar('image-tag'),

  tpuSettings+: {
    softwareVersion: "pytorch-nightly",
  },
  accelerator: tpus.v3_8,

  volumeMap+: {
    datasets: volumes.PersistentVolumeSpec {
      name: "huggingface-cluster-disk",
      mountPath: "/datasets",
    },
  },
  command: utils.scriptCommand(
    |||
      python -m pytest -s transformers/examples/pytorch/test_xla_examples.py -v
      test_exit_code=$?
      echo "\nFinished running commands.\n"
      test $test_exit_code -eq 0
    |||
  ),
};

bertBaseCased.oneshotJob
