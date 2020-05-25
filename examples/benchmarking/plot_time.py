#!/usr/bin/env python3

import csv
import math
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from transformers import HfArgumentParser


@dataclass
class PlotArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    csv_file: str = field(metadata={"help": "The csv file to plot."},)
    plot_along_batch: bool = field(
        default=False,
        metadata={"help": "Whether to plot along batch size or sequence lengh. Defaults to sequence length."},
    )
    is_time: bool = field(
        default=False,
        metadata={"help": "Whether the csv file has time results or memory results. Defaults to memory results."},
    )
    is_train: bool = field(
        default=False,
        metadata={
            "help": "Whether the csv file has training results or inference results. Defaults to inference results."
        },
    )


class Plot:
    def __init__(self, args):
        self.args = args
        self.result_dict = {}

        with open(self.args.csv_file, newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                model_name = row["model"]
                if model_name not in self.result_dict:
                    self.result_dict[model_name] = {}
                    self.result_dict[model_name]["bsz"] = []
                    self.result_dict[model_name]["seq_len"] = []
                    self.result_dict[model_name]["result"] = {}

                self.result_dict[model_name]["bsz"].append(int(row["batch_size"]))
                self.result_dict[model_name]["seq_len"].append(int(row["sequence_length"]))
                self.result_dict[model_name]["result"][(int(row["batch_size"]), int(row["sequence_length"]))] = row[
                    "result"
                ]

    def plot(self):
        fig, ax = plt.subplots()
        title_str = "Time usage" if self.args.is_time else "Memory usage"
        title_str = title_str + " for training" if self.args.is_train else title_str + " for inference"

        for model_name in self.result_dict.keys():
            batch_sizes = sorted(list(set(self.result_dict[model_name]["bsz"])))
            sequence_lengths = sorted(list(set(self.result_dict[model_name]["seq_len"])))
            results = self.result_dict[model_name]["result"]

            (x_axis_array, inner_loop_array) = (
                (batch_sizes, sequence_lengths) if self.args.plot_along_batch else (sequence_lengths, batch_sizes)
            )

            log_x_axis_array = np.asarray([math.log(x, 2) for x in x_axis_array], dtype=np.float)
            plt.xlim(min(log_x_axis_array), max(log_x_axis_array))

            for inner_loop_value in inner_loop_array:
                if self.args.plot_along_batch:
                    y_axis_array = np.asarray([results[(x, inner_loop_value)] for x in x_axis_array], dtype=np.int)
                else:
                    y_axis_array = np.asarray([results[(inner_loop_value, x)] for x in x_axis_array], dtype=np.int)

                ax.set_xscale("log", basex=2)
                ax.set_yscale("log", basey=10)

                (x_axis_label, inner_loop_label) = (
                    ("batch_size", "sequence_length in #tokens")
                    if self.args.plot_along_batch
                    else ("sequence_length in #tokens", "batch_size")
                )

                plt.scatter(
                    log_x_axis_array, y_axis_array, label=f"{model_name} - {inner_loop_label}: {inner_loop_value}"
                )
                plt.plot(log_x_axis_array, y_axis_array, "--")

            title_str += f" {model_name} vs."

        title_str = title_str[:-4]
        y_axis_label = "Time in s" if self.args.is_time else "Memory in MB"

        # plot
        plt.title(title_str)
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.legend()
        plt.show()


def main():
    parser = HfArgumentParser(PlotArguments)
    plot_args = parser.parse_args_into_dataclasses()[0]
    plot = Plot(args=plot_args)
    plot.plot()


if __name__ == "__main__":
    main()
