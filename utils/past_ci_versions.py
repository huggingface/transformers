import argparse
import os


past_versions_testing = {
    "pytorch": {
        "1.13": {
            "torch": "1.13.1",
            "torchvision": "0.14.1",
            "torchaudio": "0.13.1",
            "python": 3.9,
            "cuda": "cu116",
            "install": (
                "python3 -m pip install --no-cache-dir -U torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1"
                " --extra-index-url https://download.pytorch.org/whl/cu116"
            ),
            "base_image": "nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04",
        },
        "1.12": {
            "torch": "1.12.1",
            "torchvision": "0.13.1",
            "torchaudio": "0.12.1",
            "python": 3.9,
            "cuda": "cu113",
            "install": (
                "python3 -m pip install --no-cache-dir -U torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1"
                " --extra-index-url https://download.pytorch.org/whl/cu113"
            ),
            "base_image": "nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04",
        },
        "1.11": {
            "torch": "1.11.0",
            "torchvision": "0.12.0",
            "torchaudio": "0.11.0",
            "python": 3.9,
            "cuda": "cu113",
            "install": (
                "python3 -m pip install --no-cache-dir -U torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0"
                " --extra-index-url https://download.pytorch.org/whl/cu113"
            ),
            "base_image": "nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04",
        },
        "1.10": {
            "torch": "1.10.2",
            "torchvision": "0.11.3",
            "torchaudio": "0.10.2",
            "python": 3.9,
            "cuda": "cu113",
            "install": (
                "python3 -m pip install --no-cache-dir -U torch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2"
                " --extra-index-url https://download.pytorch.org/whl/cu113"
            ),
            "base_image": "nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04",
        },
        # torchaudio < 0.10 has no CUDA-enabled binary distributions
        "1.9": {
            "torch": "1.9.1",
            "torchvision": "0.10.1",
            "torchaudio": "0.9.1",
            "python": 3.9,
            "cuda": "cu111",
            "install": (
                "python3 -m pip install --no-cache-dir -U torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1"
                " --extra-index-url https://download.pytorch.org/whl/cu111"
            ),
            "base_image": "nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04",
        },
    },
    "tensorflow": {
        "2.11": {
            "tensorflow": "2.11.1",
            "install": "python3 -m pip install --no-cache-dir -U tensorflow==2.11.1",
            "base_image": "nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04",
        },
        "2.10": {
            "tensorflow": "2.10.1",
            "install": "python3 -m pip install --no-cache-dir -U tensorflow==2.10.1",
            "base_image": "nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04",
        },
        "2.9": {
            "tensorflow": "2.9.3",
            "install": "python3 -m pip install --no-cache-dir -U tensorflow==2.9.3",
            "base_image": "nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04",
        },
        "2.8": {
            "tensorflow": "2.8.2",
            "install": "python3 -m pip install --no-cache-dir -U tensorflow==2.8.2",
            "base_image": "nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04",
        },
        "2.7": {
            "tensorflow": "2.7.3",
            "install": "python3 -m pip install --no-cache-dir -U tensorflow==2.7.3",
            "base_image": "nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04",
        },
        "2.6": {
            "tensorflow": "2.6.5",
            "install": "python3 -m pip install --no-cache-dir -U tensorflow==2.6.5",
            "base_image": "nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04",
        },
        "2.5": {
            "tensorflow": "2.5.3",
            "install": "python3 -m pip install --no-cache-dir -U tensorflow==2.5.3",
            "base_image": "nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04",
        },
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Choose the framework and version to install")
    parser.add_argument(
        "--framework", help="The framework to install. Should be `torch` or `tensorflow`", type=str, required=True
    )
    parser.add_argument("--version", help="The version of the framework to install.", type=str, required=True)
    args = parser.parse_args()

    info = past_versions_testing[args.framework][args.version]

    os.system(f'echo "export INSTALL_CMD=\'{info["install"]}\'" >> ~/.profile')
    print(f'echo "export INSTALL_CMD=\'{info["install"]}\'" >> ~/.profile')

    cuda = ""
    if args.framework == "pytorch":
        cuda = info["cuda"]
    os.system(f"echo \"export CUDA='{cuda}'\" >> ~/.profile")
    print(f"echo \"export CUDA='{cuda}'\" >> ~/.profile")
