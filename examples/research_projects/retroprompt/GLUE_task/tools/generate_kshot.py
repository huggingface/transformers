import argparse
import os

import numpy as np
import pandas as pd


def load_datasets(data_dir, tasks):
    datasets = {}
    for task in tasks:
        # TODO: implement all GLUE tasks
        dataset = {}
        dirname = os.path.join(data_dir, task)
        if task in ["QNLI", "CoLA", "QQP", "SST-2", "RTE", "MRPC", "SNLI", "MNLI", "STS-B"]:
            if task == "MNLI":
                splits = ["train", "dev_matched", "dev_mismatched"]
            else:
                splits = ["train", "dev"]
            for split in splits:
                filename = os.path.join(dirname, f"{split}.tsv")
                with open(filename, "r") as f:
                    lines = f.readlines()
                dataset[split] = lines
            datasets[task] = dataset
        else:
            dataset = {}
            dirname = os.path.join(data_dir, task)
            splits = ["train", "test"]
            for split in splits:
                filename = os.path.join(dirname, f"{split}.csv")
                dataset[split] = pd.read_csv(filename, header=None)
            datasets[task] = dataset
    return datasets


def get_label(task, line):
    if task in ["QNLI", "CoLA", "SST-2", "QQP", "RTE", "MRPC", "SNLI", "MNLI", "STS-B"]:
        line = line.strip().split("\t")
        if task == "QNLI":
            return line[-1]
        elif task == "CoLA":
            return line[1]
        elif task == "QQP":
            return line[-1]
        elif task == "SST-2":
            return line[-1]
        elif task == "RTE":
            return line[-1]
        elif task == "MRPC":
            return line[0]
        elif task == "SNLI":
            return line[-1]
        elif task == "MNLI":
            return line[-1]
        elif task == "STS-B":
            return 0 if float(line[-1]) < 2.5 else 1
        else:
            raise NotImplementedError
    else:
        return line[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16, help="Training examples for each class.")
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        default=[
            "SST-2",
            "sst-5",
            "mr",
            "cr",
            "mpqa",
            "subj",
            "trec",
            "CoLA",
            "MRPC",
            "QQP",
            "STS-B",
            "MNLI",
            "SNLI",
            "QNLI",
            "RTE",
        ],
        help="Task names",
    )
    parser.add_argument("--seed", type=int, nargs="+", default=[13, 21, 42, 87, 100], help="Random seeds")

    parser.add_argument("--data_dir", type=str, default="data/original_data/glue", help="Path to original data")
    parser.add_argument("--output_dir", type=str, default="data/training_data", help="Output path")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, "k_shot")

    k = args.k
    print(f"K = {k}")
    datasets = load_datasets(args.data_dir, args.task)

    for seed in args.seed:
        print(f"Seed = {seed}")
        for task, dataset in datasets.items():
            np.random.seed(seed=seed)
            print(f"Task = {task}")
            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
                if task in ["CoLA"]:
                    train_header, train_lines = [], dataset["train"]
                else:
                    train_header, train_lines = dataset["train"][0:1], dataset["train"][1:]
                np.random.shuffle(train_lines)
            else:
                train_lines = dataset["train"].values.tolist()
                np.random.shuffle(train_lines)

            task_dir = os.path.join(args.output_dir, task)
            setting_dir = os.path.join(task_dir, f"{k}-{seed}")
            os.makedirs(setting_dir, exist_ok=True)

            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
                for split, lines in dataset.items():
                    if split.startswith("train"):
                        continue
                    split = split.replace("dev", "test")
                    with open(os.path.join(setting_dir, f"{split}.tsv"), "w") as f:
                        for line in lines:
                            f.write(line)
            else:
                dataset["test"].to_csv(os.path.join(setting_dir, "test.csv"), header=False, index=False)

            label_list = {}
            for line in train_lines:
                label = get_label(task, line)
                if label not in label_list:
                    label_list[label] = [line]
                else:
                    label_list[label].append(line)

            if task in ["QNLI", "CoLA", "QQP", "SST-2", "RTE", "MRPC", "SNLI", "MNLI", "STS-B"]:
                with open(os.path.join(setting_dir, "train.tsv"), "w") as f:
                    for line in train_header:
                        f.write(line)
                    for label in label_list:
                        for line in label_list[label][:k]:
                            f.write(line)
                name = "dev.tsv"
                if task == "MNLI":
                    name = "dev_matched.tsv"
                # dev_rate = 11 if '10x' in args.mode else 2
                with open(os.path.join(setting_dir, name), "w") as f:
                    for line in train_header:
                        f.write(line)
                    for label in label_list:
                        for line in label_list[label][k : k * 2]:
                            f.write(line)

                # the rest of train datsets as unlabeled data
                with open(os.path.join(setting_dir, "unlabeled.tsv"), "w") as f:
                    for line in train_header:
                        f.write(line)
                    for label in label_list:
                        for line in label_list[label][k * 2 :]:
                            f.write(line)
            else:
                new_train = []
                for label in label_list:
                    for line in label_list[label][:k]:
                        new_train.append(line)
                new_train = pd.DataFrame(new_train)
                new_train.to_csv(os.path.join(setting_dir, "train.csv"), header=False, index=False)

                new_dev = []
                for label in label_list:
                    dev_rate = 2
                    for line in label_list[label][k : k * dev_rate]:
                        new_dev.append(line)
                new_dev = pd.DataFrame(new_dev)
                new_dev.to_csv(os.path.join(setting_dir, "dev.csv"), header=False, index=False)

                unlabeled = []
                for label in label_list:
                    for line in label_list[label][k * 2 :]:
                        unlabeled.append(line)
                unlabeled = pd.DataFrame(unlabeled)
                unlabeled.to_csv(os.path.join(setting_dir, "unlabeled.csv"), header=False, index=False)


if __name__ == "__main__":
    main()
