import argparse
import os
import shutil

import timesfm

from transformers import TimesFMConfig, TimesFMModelForPrediction


"""
Sample usage:

```
python src/transformers/models/timesfm/convert_timesfm_orignal_to_pytorch.py \
    --output_dir /output/path
```
"""


def write_model(model_path, safe_serialization=True):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="cpu",
          per_core_batch_size=32,
          horizon_len=128,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
    )

    timesfm_config = TimesFMConfig(
        patch_len=tfm.hparams.input_patch_len,
        context_len=tfm.hparams.context_len,
        horizon_len=tfm.hparams.horizon_len,
        num_layers=tfm.hparams.num_layers,
        model_dim=tfm.hparams.model_dims,
        intermediate_size=tfm.hparams.model_dims,
        head_dim=tfm.hparams.model_dims//tfm.hparams.num_heads,
        num_heads=tfm.hparams.num_heads,
    )
    timesfm_config.save_pretrained(tmp_model_path)
    timesfm_model = TimesFMModelForPrediction(timesfm_config)

    # copy the weights from the original model to the new model making
    import pdb; pdb.set_trace()
    orignal_model = tfm._model


    timesfm_model.load_state_dict(tfm.state_dict())
    timesfm_model.save_pretrained(model_path, safe_serialization=safe_serialization)
    shutil.rmtree(tmp_model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument("--safe_serialization", type=bool, help="Whether or not to save using `safetensors`.")
    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        safe_serialization=args.safe_serialization,
    )

    check_outputs(args.output_dir)


if __name__ == "__main__":
    main()
