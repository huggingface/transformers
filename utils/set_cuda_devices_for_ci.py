"""A simple script to set flexibly CUDA_VISIBLE_DEVICES in GitHub Actions CI workflow files."""

import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_folder",
        type=str,
        default=None,
        help="The test folder name of the model being tested. For example, `models/cohere`.",
    )
    args = parser.parse_args()

    # `test_eager_matches_sdpa_generate` for `cohere` needs a lot of GPU memory!
    # This depends on the runners. At this moment we are targeting our AWS CI runners.
    if args.test_folder == "models/cohere":
        cuda_visible_devices = "0,1,2,3"
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    else:
        cuda_visible_devices = "0"

    print(cuda_visible_devices)
