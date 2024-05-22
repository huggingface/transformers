import argparse
import subprocess


def main(config_dir, config_name, args):
    subprocess.run(["optimum-benchmark", "--config-dir", f"{config_dir}", "--config-name", f"{config_name}"] + ["hydra/job_logging=disabled", "hydra/hydra_logging=disabled"] + args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config-dir", type=str, required=True, help="The path to the config directory.")
    parser.add_argument("--config-name", type=str, required=True, help="The config name.")
    args, unknown = parser.parse_known_args()

    main(args.config_dir, args.config_name, unknown)
