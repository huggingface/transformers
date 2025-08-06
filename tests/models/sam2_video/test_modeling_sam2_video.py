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
        # fill_hole area is set to 0 to avoid running the `get_connected_components` cuda kernel
        self.video_model = Sam2VideoModel.from_pretrained("yonigozlan/sam2.1_hiera_tiny_hf", fill_hole_area=0).to(
            torch.float32
        )
        self.processor = Sam2VideoProcessor.from_pretrained("yonigozlan/sam2.1_hiera_tiny_hf")
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
        outputs = self.video_model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            consolidate_at_video_res=False,  # Whether to save the masks at the video resolution (True) or at the model's resolution in the video session state (False)
        )
        low_res_masks = outputs.consolidated_res_masks
        video_res_masks = outputs.video_res_masks
        self.assertEqual(low_res_masks.shape, (1, 1, 256, 256))
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-21.4113, -21.4113, -22.9685], [-23.3089, -23.3089, -24.2602], [-27.5700, -27.5700, -27.1607]]
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
            frames.append(sam2_video_output.video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-21.4113, -21.4113], [-23.3089, -23.3089]]]],
                    [[[[-20.0948, -20.0948], [-21.2245, -21.2245]]]],
                    [[[[-19.9573, -19.9573], [-21.3020, -21.3020]]]],
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
            frames.append(sam2_video_output.video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-21.4113, -21.4113], [-23.3089, -23.3089]]]],
                    [[[[-20.0948, -20.0948], [-21.2245, -21.2245]]]],
                    [[[[-19.9573, -19.9573], [-21.3020, -21.3020]]]],
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
        outputs = self.video_model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            consolidate_at_video_res=False,  # Whether to save the masks at the video resolution (True) or at the model's resolution in the video session state (False)
        )
        low_res_masks = outputs.consolidated_res_masks
        video_res_masks = outputs.video_res_masks
        self.assertEqual(low_res_masks.shape, (1, 1, 256, 256))
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-11.1491, -11.1491, -11.4204], [-11.6524, -11.6524, -11.8057], [-12.7825, -12.7825, -12.6707]]
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
            frames.append(sam2_video_output.video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        # higher tolerance due to errors propagating from frame to frame
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-11.1491, -11.1491], [-11.6524, -11.6524]]]],
                    [[[[-15.3796, -15.3796], [-16.0307, -16.0307]]]],
                    [[[[-15.4860, -15.4860], [-16.4231, -16.4231]]]],
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
        outputs = self.video_model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            consolidate_at_video_res=False,  # Whether to save the masks at the video resolution (True) or at the model's resolution in the video session state (False)
        )
        low_res_masks = outputs.consolidated_res_masks
        video_res_masks = outputs.video_res_masks
        self.assertEqual(low_res_masks.shape, (1, 1, 256, 256))
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-13.1423, -13.1423, -13.6417], [-13.7748, -13.7748, -14.1142], [-15.1950, -15.1950, -15.1751]]
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
            frames.append(sam2_video_output.video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        # higher tolerance due to errors propagating from frame to frame
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-13.1423, -13.1423], [-13.7748, -13.7748]]]],
                    [[[[-14.9971, -14.9971], [-15.7066, -15.7066]]]],
                    [[[[-15.4576, -15.4576], [-16.1667, -16.1667]]]],
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
        outputs = self.video_model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            consolidate_at_video_res=False,  # Whether to save the masks at the video resolution (True) or at the model's resolution in the video session state (False)
        )
        low_res_masks = outputs.consolidated_res_masks
        video_res_masks = outputs.video_res_masks
        self.assertEqual(low_res_masks.shape, (1, 1, 256, 256))
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-12.3523, -12.3523, -12.8905], [-13.0603, -13.0603, -13.4075], [-14.6503, -14.6503, -14.5686]]
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
            frames.append(sam2_video_output.video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        # higher tolerance due to errors propagating from frame to frame
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-12.3523, -12.3523], [-13.0603, -13.0603]]]],
                    [[[[-15.8179, -15.8179], [-16.4159, -16.4159]]]],
                    [[[[-15.8949, -15.8949], [-16.6002, -16.6002]]]],
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
        outputs = self.video_model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            consolidate_at_video_res=False,  # Whether to save the masks at the video resolution (True) or at the model's resolution in the video session state (False)
        )
        low_res_masks = outputs.consolidated_res_masks
        video_res_masks = outputs.video_res_masks
        self.assertEqual(low_res_masks.shape, (2, 1, 256, 256))
        self.assertEqual(video_res_masks.shape, (2, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[:, 0, :2, :2],  # first object
            torch.tensor(
                [[[-12.6303, -12.6303], [-13.3667, -13.3667]], [[-20.3307, -20.3307], [-22.0473, -22.0473]]]
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
            frames.append(sam2_video_output.video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 2, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-12.6303, -12.6303], [-13.3667, -13.3667]]], [[[-20.3307, -20.3307], [-22.0473, -22.0473]]]],
                    [[[[-18.5245, -18.5245], [-19.5829, -19.5829]]], [[[-17.5492, -17.5492], [-19.2210, -19.2210]]]],
                    [[[[-14.2722, -14.2722], [-15.4622, -15.4622]]], [[[-18.3148, -18.3148], [-20.0278, -20.0278]]]],
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
        sam2_video_output = self.video_model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            consolidate_at_video_res=True,  # Whether to save the masks at the video resolution (True) or at the model's resolution in the video session state (False)
        )

        # set mask as input
        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_id,
            input_masks=sam2_video_output.video_res_masks,
        )
        sam2_video_output = self.video_model(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            consolidate_at_video_res=False,  # Whether to save the masks at the video resolution (True) or at the model's resolution in the video session state (False)
        )
        low_res_masks = sam2_video_output.consolidated_res_masks
        video_res_masks = sam2_video_output.video_res_masks
        self.assertEqual(low_res_masks.shape, (1, 1, 256, 256))
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
            frames.append(sam2_video_output.video_res_masks)
        frames = torch.stack(frames, dim=0)
        self.assertEqual(frames.shape, (3, 1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            frames[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-10.0000, -10.0000], [-10.0000, -10.0000]]]],
                    [[[[-18.3645, -18.3645], [-19.2324, -19.2324]]]],
                    [[[[-20.3382, -20.3382], [-21.1854, -21.1854]]]],
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
            video_res_masks.append(sam2_video_output.video_res_masks)

        video_res_masks = torch.stack(video_res_masks, dim=0)
        self.assertEqual(
            video_res_masks.shape, (max_frame_num_to_track, 1, 1, raw_video.shape[-3], raw_video.shape[-2])
        )
        # higher tolerance due to errors propagating from frame to frame
        torch.testing.assert_close(
            video_res_masks[:3, :, :, :2, :2],
            torch.tensor(
                [
                    [[[[-11.1491, -11.1491], [-11.6524, -11.6524]]]],
                    [[[[-15.3796, -15.3796], [-16.0307, -16.0307]]]],
                    [[[[-15.4860, -15.4860], [-16.4231, -16.4231]]]],
                ]
            ).to(torch_device),
            atol=1e-2,
            rtol=1e-2,
        )
