import getopt
import json
import os

# import numpy as np
import sys
from collections import OrderedDict

import datasets
import numpy as np
import torch
from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess

from utils import Config


"""
USAGE:
``python extracting_data.py -i <img_dir> -o <dataset_file>.datasets <batch_size>``
"""


TEST = False
CONFIG = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
DEFAULT_SCHEMA = datasets.Features(
    OrderedDict(
        {
            "attr_ids": datasets.Sequence(length=CONFIG.MAX_DETECTIONS, feature=datasets.Value("float32")),
            "attr_probs": datasets.Sequence(length=CONFIG.MAX_DETECTIONS, feature=datasets.Value("float32")),
            "boxes": datasets.Array2D((CONFIG.MAX_DETECTIONS, 4), dtype="float32"),
            "img_id": datasets.Value("int32"),
            "obj_ids": datasets.Sequence(length=CONFIG.MAX_DETECTIONS, feature=datasets.Value("float32")),
            "obj_probs": datasets.Sequence(length=CONFIG.MAX_DETECTIONS, feature=datasets.Value("float32")),
            "roi_features": datasets.Array2D((CONFIG.MAX_DETECTIONS, 2048), dtype="float32"),
            "sizes": datasets.Sequence(length=2, feature=datasets.Value("float32")),
            "preds_per_image": datasets.Value(dtype="int32"),
        }
    )
)


class Extract:
    def __init__(self, argv=sys.argv[1:]):
        inputdir = None
        outputfile = None
        subset_list = None
        batch_size = 1
        opts, args = getopt.getopt(argv, "i:o:b:s", ["inputdir=", "outfile=", "batch_size=", "subset_list="])
        for opt, arg in opts:
            if opt in ("-i", "--inputdir"):
                inputdir = arg
            elif opt in ("-o", "--outfile"):
                outputfile = arg
            elif opt in ("-b", "--batch_size"):
                batch_size = int(arg)
            elif opt in ("-s", "--subset_list"):
                subset_list = arg

        assert inputdir is not None  # and os.path.isdir(inputdir), f"{inputdir}"
        assert outputfile is not None and not os.path.isfile(outputfile), f"{outputfile}"
        if subset_list is not None:
            with open(os.path.realpath(subset_list)) as f:
                self.subset_list = {self._vqa_file_split()[0] for x in tryload(f)}
        else:
            self.subset_list = None

        self.config = CONFIG
        if torch.cuda.is_available():
            self.config.model.device = "cuda"
        self.inputdir = os.path.realpath(inputdir)
        self.outputfile = os.path.realpath(outputfile)
        self.preprocess = Preprocess(self.config)
        self.model = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.config)
        self.batch = batch_size if batch_size != 0 else 1
        self.schema = DEFAULT_SCHEMA

    def _vqa_file_split(self, file):
        img_id = int(file.split(".")[0].split("_")[-1])
        filepath = os.path.join(self.inputdir, file)
        return (img_id, filepath)

    @property
    def file_generator(self):
        batch = []
        for i, file in enumerate(os.listdir(self.inputdir)):
            if self.subset_list is not None and i not in self.subset_list:
                continue
            batch.append(self._vqa_file_split(file))
            if len(batch) == self.batch:
                temp = batch
                batch = []
                yield list(map(list, zip(*temp)))

        for i in range(1):
            yield list(map(list, zip(*batch)))

    def __call__(self):
        # make writer
        if not TEST:
            writer = datasets.ArrowWriter(features=self.schema, path=self.outputfile)
        # do file generator
        for i, (img_ids, filepaths) in enumerate(self.file_generator):
            images, sizes, scales_yx = self.preprocess(filepaths)
            output_dict = self.model(
                images,
                sizes,
                scales_yx=scales_yx,
                padding="max_detections",
                max_detections=self.config.MAX_DETECTIONS,
                pad_value=0,
                return_tensors="np",
                location="cpu",
            )
            output_dict["boxes"] = output_dict.pop("normalized_boxes")
            if not TEST:
                output_dict["img_id"] = np.array(img_ids)
                batch = self.schema.encode_batch(output_dict)
                writer.write_batch(batch)
            if TEST:
                break
            # finalizer the writer
        if not TEST:
            num_examples, num_bytes = writer.finalize()
            print(f"Success! You wrote {num_examples} entry(s) and {num_bytes >> 20} mb")


def tryload(stream):
    try:
        data = json.load(stream)
        try:
            data = list(data.keys())
        except Exception:
            data = [d["img_id"] for d in data]
    except Exception:
        try:
            data = eval(stream.read())
        except Exception:
            data = stream.read().split("\n")
    return data


if __name__ == "__main__":
    extract = Extract(sys.argv[1:])
    extract()
    if not TEST:
        dataset = datasets.Dataset.from_file(extract.outputfile)
        # wala!
        # print(np.array(dataset[0:2]["roi_features"]).shape)
