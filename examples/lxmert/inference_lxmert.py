"""
 coding=utf-8
 Copyright 2018, Antonio Mendoza Hao Tan, Mohit Bansal
 Adapted From Facebook Inc, Detectron2

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.import copy
 """

import copy
import json
import pickle as pkl
from collections import OrderedDict

import numpy as np
import PIL.Image as Image
import torch
from yaml import Loader, dump, load

from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess, tensorize
from transformers import LxmertForQuestionAnswering, LxmertTokenizer
from visualizing_image import SingleImageViz


def load_config(config="config.yaml"):
    with open(config) as stream:
        data = load(stream, Loader=Loader)
    return Config(data)


def load_obj_data(objs="objects.txt", attrs="attributes.txt"):
    vg_classes = []
    with open(objs) as f:
        for object in f.readlines():
            vg_classes.append(object.split(",")[0].lower().strip())

    vg_attrs = []
    with open(attrs) as f:
        for object in f.readlines():
            vg_attrs.append(object.split(",")[0].lower().strip())
    return vg_classes, vg_attrs


def load_ckp(ckp="checkpoint.pkl"):
    r = OrderedDict()
    with open(ckp, "rb") as f:
        ckp = pkl.load(f)["model"]
    for k in copy.deepcopy(list(ckp.keys())):
        v = ckp.pop(k)
        if isinstance(v, np.ndarray):
            v = torch.tensor(v)
        else:
            assert isinstance(v, torch.tensor), type(v)
        r[k] = v
    return r


def show_image(a):
    a = np.uint8(np.clip(a, 0, 255))
    img = Image.fromarray(a)
    img.show()


def save_image(a, name="test_out", affix="jpg"):
    a = np.uint8(np.clipk(a, 0, 255))
    img = Image.fromarray(a)
    img.save(f"{name}.{affix}")


class Config:
    def __init__(self, dictionary: dict, name: str = "root", level=0):
        self._name = name
        self._level = level
        d = {}
        for k, v in dictionary.items():
            if v is None:
                raise ValueError()
            k = copy.deepcopy(k)
            v = copy.deepcopy(v)
            if isinstance(v, dict):
                v = Config(v, name=k, level=level + 1)
            d[k] = v
            setattr(self, k, v)
            setattr(self, k.upper(), getattr(self, k))

        self._pointer = d

    def __repr__(self):
        return str(list((self._pointer.keys())))

    def to_dict(self):
        return self._pointer

    def dump_yaml(self, data, file_name):
        with open(f"{file_name}", "w") as stream:
            dump(data, stream)

    def dump_json(self, data, file_name):
        with open(f"{file_name}", "w") as stream:
            json.dump(data, stream)

    def __str__(self):
        t = "  "
        r = f"{t * (self._level)}{self._name.upper()}:\n"
        level = self._level
        for i, (k, v) in enumerate(self._pointer.items()):
            if isinstance(v, Config):
                r += f"{t * (self._level)}{v}\n"
                self._level += 1
            else:
                r += f"{t * (self._level + 1)}{k}:{v}({type(v).__name__})\n"
            self._level = level
        return r[:-1]


if __name__ == "__main__":
    im1 = input("type the path of the image you want to process: ")
    test_question = input("type the question you would like to ask about the image: ")
    # incase I want to batch
    img_tensors = list(map(lambda x: tensorize(x), [im1]))
    cfg = load_config()
    objids, attrids = load_obj_data()
    gqa_answers = json.load(open("gqa_answers.json"))
    # init classes
    visualizer = SingleImageViz(img_tensors[0], id2obj=objids, id2attr=attrids)
    preprocess = Preprocess(cfg)
    frcnn = GeneralizedRCNN(cfg)
    frcnn.load_state_dict(load_ckp(), strict=False)
    frcnn.eval()
    lxmert_tokenizer = LxmertTokenizer.from_pretrained("eltoto1219/lxmert-gqa-untuned")
    lxmert = LxmertForQuestionAnswering.from_pretrained(
        "eltoto1219/lxmert-gqa-untuned", num_qa_labels=len(gqa_answers)
    )
    lxmert.eval()
    # run frcnn
    images, sizes, scales_yx = preprocess(img_tensors)
    output_dict = frcnn(images, sizes, scales_yx=scales_yx)
    # only want to select the first image
    output_dict = output_dict[0]
    features = output_dict.pop("roi_features")
    boxes = output_dict.pop("boxes")
    # add boxes and labels to the image
    visualizer.draw_boxes(
        boxes,
        output_dict.pop("obj_ids"),
        output_dict.pop("obj_scores"),
        output_dict.pop("attr_ids"),
        output_dict.pop("attr_scores"),
    )
    visualizer.save()
    # run lxmert
    inputs = lxmert_tokenizer(
        test_question,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
    )
    input_ids = torch.tensor(inputs.input_ids)
    output = lxmert(
        input_ids=input_ids,
        attention_mask=torch.tensor(inputs.attention_mask),
        visual_feats=features.unsqueeze(0),
        visual_pos=boxes.unsqueeze(0),
        token_type_ids=torch.tensor(inputs.token_type_ids),
        return_dict=True,
        output_attentions=False,
    )
    logit, pred = output["question_answering_score"].max(-1)
    print("prediction:", gqa_answers[pred])
    print("class ind:", int(pred))
