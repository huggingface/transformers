import csv
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from transformers import HfArgumentParser


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class PlotArguments:
    """
    Arguments pertaining to plotting model performance data from a CSV file.
    """

    csv_file: str = field(
        metadata={"help": "The CSV file to plot."},
    )
    plot_along_batch: bool = field(
        default=False,
        metadata={"help": "Whether to plot along batch size or sequence length. Defaults to sequence length."},
    )
    is_time: bool = field(
        default=False,
        metadata={"help": "Whether the CSV file contains time results. Defaults to memory results."},
    )
    no_log_scale: bool = field(
        default=False,
        metadata={"help": "Disable logarithmic scale when plotting"},
    )
    is_train: bool = field(
        default=False,
        metadata={"help": "Whether the CSV file contains training results. Defaults to inference results."},
    )
    figure_png_file: Optional[str] = field(
        default=None,
        metadata={"help": "Filename to save the plot. If not specified, the plot will be displayed but not saved."},
    )
    short_model_names: Optional[List[str]] = list_field(
        default=None, metadata={"help": "List of model names to use in the plot."}
    )


def can_convert_to_int(string):
    """Check if a string can be converted to an integer."""
    try:
        int(string)
        return True
    except ValueError:
        return False


def can_convert_to_float(string):
    """Check if a string can be converted to a float."""
    try:
        float(string)
        return True
    except ValueError:
        return False


class Plot:
    def __init__(self, args):
        self.args = args
        self.result_dict = defaultdict(lambda: {"bsz": [], "seq_len": [], "result": {}})

        try:
            with open(self.args.csv_file, newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    model_name = row["model"]
                    batch_size = int(row["batch_size"])
                    seq_len = int(row["sequence_length"])
                    result = row["result"]

                    self.result_dict[model_name]["bsz"].append(batch_size)
                    self.result_dict[model_name]["seq_len"].append(seq_len)

                    if can_convert_to_int(result):
                        self.result_dict[model_name]["result"][(batch_size, seq_len)] = int(result)
                    elif can_convert_to_float(result):
                        self.result_dict[model_name]["result"][(batch_size, seq_len)] = float(result)
                    else:
                        print(f"Warning: Unrecognized result format in row: {row}")

        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {self.args.csv_file}")
        except Exception as e:
            print(f"Error reading the CSV file: {e}")

    def plot(self):
        fig, ax = plt.subplots()
        title_str = "Time usage" if self.args.is_time else "Memory usage"
        title_str += " for training" if self.args.is_train else " for inference"

        if not self.args.no_log_scale:
            # Apply logarithmic scales if data is non-zero
            ax.set_xscale("log")
            ax.set_yscale("log")

        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())

        for model_name_idx, model_name in enumerate(self.result_dict.keys()):
            batch_sizes = sorted(set(self.result_dict[model_name]["bsz"]))
            sequence_lengths = sorted(set(self.result_dict[model_name]["seq_len"]))
            results = self.result_dict[model_name]["result"]

            (x_axis_array, inner_loop_array) = (
                (batch_sizes, sequence_lengths) if self.args.plot_along_batch else (sequence_lengths, batch_sizes)
            )

            label_model_name = (
                model_name if self.args.short_model_names is None else self.args.short_model_names[model_name_idx]
            )

            for inner_loop_value in inner_loop_array:
                try:
                    if self.args.plot_along_batch:
                        y_axis_array = np.asarray(
                            [results[(x, inner_loop_value)] for x in x_axis_array if (x, inner_loop_value) in results],
                            dtype=float,
                        )
                    else:
                        y_axis_array = np.asarray(
                            [results[(inner_loop_value, x)] for x in x_axis_array if (inner_loop_value, x) in results],
                            dtype=float,
                        )

                    x_axis_array = np.asarray(x_axis_array, dtype=float)[: len(y_axis_array)]
                    plt.scatter(
                        x_axis_array, y_axis_array, label=f"{label_model_name} - {inner_loop_label}: {inner_loop_value}"
                    )
                    plt.plot(x_axis_array, y_axis_array, "--")

                except Exception as e:
                    print(f"Error plotting data for model {model_name}: {e}")

            title_str += f" {label_model_name} vs."

        title_str = title_str.rstrip(" vs.")
        y_axis_label = "Time in s" if self.args.is_time else "Memory in MB"

        plt.title(title_str)
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.legend()

        if self.args.figure_png_file is not None:
            try:
                plt.savefig(self.args.figure_png_file)
            except Exception as e:
                print(f"Error saving the plot: {e}")
        else:
            plt.show()


def main():
    parser = HfArgumentParser(PlotArguments)
    plot_args = parser.parse_args_into_dataclasses()[0]
    print(f"Plot arguments: {plot_args}")  # Debug: Print plot arguments
    
    try:
        plot = Plot(args=plot_args)
        plot.plot()
    except Exception as e:
        print(f"Error: {e}")  # Debug: Catch and print any errors


if __name__ == "__main__":
    main()
