# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the Wav2Vec2 tokenizer."""

import inspect
import json
import os
import random
import shutil
import tempfile
import unittest

import torch
import numpy as np

from PIL import Image
import requests

from transformers.models.detr.tokenization_detr import DetrTokenizer

class DetrTokenizerTest(unittest.TestCase):
    tokenizer_class = DetrTokenizer

    def setUp(self):
        super().setUp()
        
        # single PIL image
        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        self.img = Image.open(requests.get(url, stream=True).raw)

        # batch of PIL images + annotations
        base_url = "http://images.cocodataset.org/val2017/"
        image_urls = ["000000087038.jpg", "000000578500.jpg", "000000261982.jpg"] 

        images = []
        for image_url in image_urls:
            images.append(Image.open(requests.get(base_url + image_url, stream=True).raw))
        self.images = images

        # each target is a dict with keys "boxes" and "area"
        self.annotations = [{'boxes': [[253.21, 271.07, 59.59, 60.97],
                                    [226.04, 229.31, 11.59, 30.41],
                                    [257.85, 224.48, 44.13, 97.0],
                                    [68.18, 238.19, 16.18, 42.88],
                                    [79.16, 232.26, 28.22, 51.12],
                                    [98.4, 234.28, 19.52, 46.46],
                                    [326.86, 223.46, 13.11, 38.67],
                                    [155.27, 246.34, 14.87, 21.99],
                                    [298.61, 316.85, 63.91, 47.19],
                                    [345.41, 173.41, 72.94, 185.41],
                                    [239.72, 225.38, 10.64, 33.06],
                                    [167.02, 234.0, 15.78, 37.46],
                                    [209.68, 231.08, 9.15, 34.53],
                                    [408.29, 231.25, 17.12, 34.97],
                                    [204.14, 229.02, 7.33, 34.96],
                                    [195.32, 228.06, 10.65, 37.18],
                                    [1, 190, 638, 101]],
                             'area': [1391.4269500000005, 
                                        232.4970999999999, 
                                        1683.128300000001, 
                                        413.4482999999996, 
                                        563.41615, 
                                        363.24569999999994, 
                                        261.10905000000054, 
                                        152.3124499999996, 
                                        1268.7499999999989, 
                                        4686.905750000002, 
                                        204.17735000000013, 
                                        277.0192999999997, 
                                        241.29070000000024, 
                                        243.31384999999952, 
                                        188.82489999999987, 
                                        294.38859999999977, 
                                        6443],
                            },
                            {'boxes': [[268.66, 143.28, 61.01, 53.52],
                                        [204.04, 139.88, 24.87, 36.95],
                                        [157.51, 135.92, 26.51, 54.95],
                                        [117.06, 135.86, 37.88, 56.7],
                                        [192.39, 137.85, 14.19, 46.86],
                                        [311.46, 149.17, 156.95, 88.98],
                                        [499.59, 116.56, 140.41, 173.44],
                                        [1.86, 147.85, 132.27, 99.36],
                                        [124.21, 150.01, 89.08, 5.36],
                                        [344.97, 92.63, 6.72, 30.25],
                                        [441.77, 71.9, 10.01, 41.52],
                                        [118.63, 153.62, 8.78, 20.32],
                                        [291.45, 179.46, 15.1, 10.3],
                                        [498.7, 115.61, 141.3, 174.39]],
                            },
                            {'boxes': [[0.0, 8.53, 251.9, 214.62],
                                        [409.89, 120.81, 47.11, 92.04],
                                        [84.85, 0.0, 298.6, 398.22],
                                        [159.71, 211.53, 189.89, 231.01],
                                        [357.69, 110.26, 99.31, 90.43]],
                            }
        ]

    def get_tokenizer(self, **kwars):
        return DetrTokenizer()

    # tests on single PIL image (inference only)
    def test_tokenizer(self):
        tokenizer = self.get_tokenizer()
        encoding = tokenizer(self.img)
        
        self.assertEqual(encoding["pixel_values"].shape, (1, 3, 800, 1066))
        self.assertEqual(encoding["pixel_mask"].shape, (1, 800, 1066))

    # tests on single PIL image (inference only, with resize set to False)
    def test_tokenizer_no_resize(self):
        tokenizer = self.get_tokenizer()
        encoding = tokenizer(self.img, resize=False)

        self.assertEqual(encoding["pixel_values"].shape, (1, 3, 480, 640))
        self.assertEqual(encoding["pixel_mask"].shape, (1, 480, 640))

    # tests on batch of PIL images (inference only)
    def test_tokenizer_batch(self):
        tokenizer = self.get_tokenizer()
        encoding = tokenizer(self.images)

        self.assertEqual(encoding["pixel_values"].shape, (3, 3, 1120, 1332))
        self.assertEqual(encoding["pixel_mask"].shape, (3, 1120, 1332))

    # tests on batch of PIL images (training, i.e. with annotations)
    def test_tokenizer_batch_training(self):
        tokenizer = self.get_tokenizer()
        encoding = tokenizer(self.images, self.annotations)

        self.assertEqual(encoding["pixel_values"].shape, (3, 3, 1120, 1332))
        self.assertEqual(encoding["pixel_mask"].shape, (3, 1120, 1332))

