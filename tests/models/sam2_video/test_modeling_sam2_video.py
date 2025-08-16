# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch SAM2 model."""

import gc
import unittest

import requests

from transformers.testing_utils import (
    backend_empty_cache,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available
from transformers.video_utils import load_video


if is_torch_available():
    import torch

    from transformers import Sam2VideoModel, Sam2VideoProcessor


if is_vision_available():
    from PIL import Image


def prepare_image():
    img_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image


def prepare_groceries_image():
    img_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/groceries.jpg"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image


def prepare_dog_img():
    img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dog-sam.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    return raw_image


def prepare_video():
    video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
    raw_video, _ = load_video(video_url)
    return raw_video


@slow
class Sam2VideoModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.video_model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-tiny").to(torch.float32)
        self.processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-tiny")
        self.video_model.to(torch_device)
        self.video_model.eval()

    def tearDown(self):
        super().tearDown()
        # clean-up as much as possible GPU memory occupied by PyTorch
        gc.collect()
        backend_empty_cache(torch_device)

    def test_inference_mask_generation_video_one_point(self):
        raw_video = prepare_video()
        inference_session = self.processor.init_video_session(video=raw_video, inference_device=torch_device)
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_points=[[[[210, 350]]]],
            input_labels=[[[1]]],
        )
        outputs = self.video_model(inference_session=inference_session, frame_idx=ann_frame_idx)
        low_res_masks = outputs.pred_masks
        self.assertEqual(low_res_masks.shape, (1, 1, 256, 256))
        video_res_masks = self.processor.post_process_masks([low_res_masks], [raw_video.shape[-3:-1]], binarize=False)[
            0
        ]
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-21.4113, -21.4113, -22.9687], [-23.3090, -23.3090, -24.2606], [-27.5705, -27.5705, -27.1616]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        # test propagate in video frames
        frames = []
        for sam2_video_output in self.video_model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=2,
        ):
            video_res_masks = self.processor.post_process_masks(
                [sam2_video_output.pred_masks], [raw_video.shape[-3:-1]], binarize=False
            )[0]
            frames.append(video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-21.4113, -21.4113], [-23.3090, -23.3090]]]],
                    [[[[-20.1003, -20.1003], [-21.2294, -21.2294]]]],
                    [[[[-19.9619, -19.9619], [-21.3060, -21.3060]]]],
                ],
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_mask_generation_video_one_point_propagate_in_video_directly(self):
        raw_video = prepare_video()
        inference_session = self.processor.init_video_session(video=raw_video, inference_device=torch_device)
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_points=[[[[210, 350]]]],
            input_labels=[[[1]]],
        )
        # test propagate in video frames
        frames = []
        for sam2_video_output in self.video_model.propagate_in_video_iterator(
            inference_session=inference_session,
            start_frame_idx=ann_frame_idx,
            max_frame_num_to_track=2,
        ):
            video_res_masks = self.processor.post_process_masks(
                [sam2_video_output.pred_masks], [raw_video.shape[-3:-1]], binarize=False
            )[0]
            frames.append(video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-21.4113, -21.4113], [-23.3090, -23.3090]]]],
                    [[[[-20.1003, -20.1003], [-21.2294, -21.2294]]]],
                    [[[[-19.9619, -19.9619], [-21.3060, -21.3060]]]],
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_mask_generation_video_multi_points(self):
        raw_video = prepare_video()
        inference_session = self.processor.init_video_session(video=raw_video, inference_device=torch_device)
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_points=[[[[210, 350], [250, 220]]]],
            input_labels=[[[1, 1]]],
        )
        outputs = self.video_model(inference_session=inference_session, frame_idx=ann_frame_idx)
        low_res_masks = outputs.pred_masks
        video_res_masks = self.processor.post_process_masks(
            [outputs.pred_masks], [raw_video.shape[-3:-1]], binarize=False
        )[0]
        self.assertEqual(low_res_masks.shape, (1, 1, 256, 256))
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-11.1487, -11.1487, -11.4202], [-11.6522, -11.6522, -11.8057], [-12.7829, -12.7829, -12.6715]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        # test propagate in video frames
        frames = []
        for sam2_video_output in self.video_model.propagate_in_video_iterator(
            inference_session=inference_session,
            start_frame_idx=ann_frame_idx,
            max_frame_num_to_track=2,
        ):
            video_res_masks = self.processor.post_process_masks(
                [sam2_video_output.pred_masks], [raw_video.shape[-3:-1]], binarize=False
            )[0]
            frames.append(video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        # higher tolerance due to errors propagating from frame to frame
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-11.1487, -11.1487], [-11.6522, -11.6522]]]],
                    [[[[-15.3821, -15.3821], [-16.0333, -16.0333]]]],
                    [[[[-15.4855, -15.4855], [-16.4230, -16.4230]]]],
                ]
            ).to(torch_device),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_inference_mask_generation_video_one_bb(self):
        raw_video = prepare_video()
        inference_session = self.processor.init_video_session(video=raw_video, inference_device=torch_device)
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_boxes=[[[300, 0, 500, 400]]],
        )
        outputs = self.video_model(inference_session=inference_session, frame_idx=ann_frame_idx)
        low_res_masks = outputs.pred_masks
        video_res_masks = self.processor.post_process_masks(
            [outputs.pred_masks], [raw_video.shape[-3:-1]], binarize=False
        )[0]
        self.assertEqual(low_res_masks.shape, (1, 1, 256, 256))
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-13.1427, -13.1427, -13.6418], [-13.7753, -13.7753, -14.1144], [-15.1957, -15.1957, -15.1757]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        # test propagate in video frames
        frames = []
        for sam2_video_output in self.video_model.propagate_in_video_iterator(
            inference_session=inference_session,
            start_frame_idx=ann_frame_idx,
            max_frame_num_to_track=2,
        ):
            video_res_masks = self.processor.post_process_masks(
                [sam2_video_output.pred_masks], [raw_video.shape[-3:-1]], binarize=False
            )[0]
            frames.append(video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        # higher tolerance due to errors propagating from frame to frame
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-13.1427, -13.1427], [-13.7753, -13.7753]]]],
                    [[[[-14.9998, -14.9998], [-15.7086, -15.7086]]]],
                    [[[[-15.4558, -15.4558], [-16.1649, -16.1649]]]],
                ]
            ).to(torch_device),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_inference_mask_generation_video_one_point_one_bb(self):
        raw_video = prepare_video()
        inference_session = self.processor.init_video_session(video=raw_video, inference_device=torch_device)
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_boxes=[[[300, 0, 500, 400]]],
            input_points=[[[[460, 60]]]],
            input_labels=[[[1]]],
        )
        outputs = self.video_model(inference_session=inference_session, frame_idx=ann_frame_idx)
        low_res_masks = outputs.pred_masks
        video_res_masks = self.processor.post_process_masks(
            [outputs.pred_masks], [raw_video.shape[-3:-1]], binarize=False
        )[0]
        self.assertEqual(low_res_masks.shape, (1, 1, 256, 256))
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-12.3525, -12.3525, -12.8907], [-13.0608, -13.0608, -13.4079], [-14.6511, -14.6511, -14.5694]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        # test propagate in video frames
        frames = []
        for sam2_video_output in self.video_model.propagate_in_video_iterator(
            inference_session=inference_session,
            start_frame_idx=ann_frame_idx,
            max_frame_num_to_track=2,
        ):
            video_res_masks = self.processor.post_process_masks(
                [sam2_video_output.pred_masks], [raw_video.shape[-3:-1]], binarize=False
            )[0]
            frames.append(video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        # higher tolerance due to errors propagating from frame to frame
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-12.3525, -12.3525], [-13.0608, -13.0608]]]],
                    [[[[-15.8181, -15.8181], [-16.4163, -16.4163]]]],
                    [[[[-15.8900, -15.8900], [-16.5953, -16.5953]]]],
                ]
            ).to(torch_device),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_inference_mask_generation_video_multi_objects_multi_points(self):
        raw_video = prepare_video()
        inference_session = self.processor.init_video_session(video=raw_video, inference_device=torch_device)
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_ids = [2, 3]  # give a unique id to each object we interact with (it can be any integers)

        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_ids,
            input_points=[[[[200, 300], [230, 250], [275, 175]], [[400, 150]]]],
            input_labels=[[[1, 1, 0], [1]]],
        )
        outputs = self.video_model(inference_session=inference_session, frame_idx=ann_frame_idx)
        low_res_masks = outputs.pred_masks
        video_res_masks = self.processor.post_process_masks(
            [outputs.pred_masks], [raw_video.shape[-3:-1]], binarize=False
        )[0]
        self.assertEqual(low_res_masks.shape, (2, 1, 256, 256))
        self.assertEqual(video_res_masks.shape, (2, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[:, 0, :2, :2],  # first object
            torch.tensor(
                [[[-12.6294, -12.6294], [-13.3659, -13.3659]], [[-20.3319, -20.3319], [-22.0491, -22.0491]]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        # test propagate in video frames
        frames = []
        for sam2_video_output in self.video_model.propagate_in_video_iterator(
            inference_session=inference_session,
            start_frame_idx=ann_frame_idx,
            max_frame_num_to_track=2,
        ):
            video_res_masks = self.processor.post_process_masks(
                [sam2_video_output.pred_masks], [raw_video.shape[-3:-1]], binarize=False
            )[0]
            frames.append(video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 2, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-12.6294, -12.6294], [-13.3659, -13.3659]]], [[[-20.3319, -20.3319], [-22.0491, -22.0491]]]],
                    [[[[-18.5249, -18.5249], [-19.5830, -19.5830]]], [[[-17.5537, -17.5537], [-19.2259, -19.2259]]]],
                    [[[[-14.2722, -14.2722], [-15.4622, -15.4622]]], [[[-18.3185, -18.3185], [-20.0314, -20.0314]]]],
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_propagate_video_from_mask_input(self):
        raw_video = prepare_video()
        inference_session = self.processor.init_video_session(video=raw_video, inference_device=torch_device)
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        # get input_mask
        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_points=[[[[210, 350], [250, 220]]]],
            input_labels=[[[1, 1]]],
        )
        sam2_video_output = self.video_model(inference_session=inference_session, frame_idx=ann_frame_idx)

        # set mask as input
        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_masks=self.processor.post_process_masks(
                [sam2_video_output.pred_masks], [raw_video.shape[-3:-1]], binarize=False
            )[0],
        )
        sam2_video_output = self.video_model(inference_session=inference_session, frame_idx=ann_frame_idx)
        low_res_masks = sam2_video_output.pred_masks
        self.assertEqual(low_res_masks.shape, (1, 1, 256, 256))
        video_res_masks = self.processor.post_process_masks(
            [sam2_video_output.pred_masks], [raw_video.shape[-3:-1]], binarize=False
        )[0]
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-10.0000, -10.0000, -10.0000], [-10.0000, -10.0000, -10.0000], [-10.0000, -10.0000, -10.0000]]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

        # test propagate in video frames
        frames = []
        for sam2_video_output in self.video_model.propagate_in_video_iterator(
            inference_session=inference_session,
            start_frame_idx=ann_frame_idx,
            max_frame_num_to_track=2,
        ):
            video_res_masks = self.processor.post_process_masks(
                [sam2_video_output.pred_masks], [raw_video.shape[-3:-1]], binarize=False
            )[0]
            frames.append(video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-10.0000, -10.0000], [-10.0000, -10.0000]]]],
                    [[[[-18.4807, -18.4807], [-19.1966, -19.1966]]]],
                    [[[[-20.0512, -20.0512], [-20.9110, -20.9110]]]],
                ],
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_propagate_on_streamed_video(self):
        raw_video = prepare_video()

        inference_session = self.processor.init_video_session(inference_device=torch_device)
        video_res_masks = []
        max_frame_num_to_track = 3
        for frame_idx, frame in enumerate(raw_video):
            if frame_idx >= max_frame_num_to_track:
                break
            inputs = self.processor(images=frame, device=torch_device, return_tensors="pt")
            if frame_idx == 0:
                self.processor.add_inputs_to_inference_session(
                    inference_session,
                    frame_idx=0,
                    obj_ids=1,
                    input_points=[[[[210, 350], [250, 220]]]],
                    input_labels=[[[1, 1]]],
                    original_size=inputs.original_sizes[0],
                )
            sam2_video_output = self.video_model(inference_session=inference_session, frame=inputs.pixel_values[0])
            video_res_masks.append(
                self.processor.post_process_masks(
                    [sam2_video_output.pred_masks], inputs.original_sizes, binarize=False
                )[0]
            )

        video_res_masks = torch.stack(video_res_masks, dim=0)
        self.assertEqual(
            video_res_masks.shape, (max_frame_num_to_track, 1, 1, raw_video.shape[-3], raw_video.shape[-2])
        )
        # higher tolerance due to errors propagating from frame to frame
        torch.testing.assert_close(
            video_res_masks[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-11.1487, -11.1487], [-11.6522, -11.6522]]]],
                    [[[[-15.3821, -15.3821], [-16.0333, -16.0333]]]],
                    [[[[-15.4855, -15.4855], [-16.4230, -16.4230]]]],
                ]
            ).to(torch_device),
            atol=1e-2,
            rtol=1e-2,
        )
