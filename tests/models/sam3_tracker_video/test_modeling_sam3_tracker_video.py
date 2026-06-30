# Copyright 2025 the HuggingFace Team. All rights reserved.
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
    is_torch_bf16_available_on_device,
    is_torch_fp16_available_on_device,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available
from transformers.video_utils import load_video


if is_torch_available():
    import torch

    from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor


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
class Sam3TrackerVideoModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.video_model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(torch.float32)
        self.processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
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
        self.assertEqual(low_res_masks.shape, (1, 1, 288, 288))
        video_res_masks = self.processor.post_process_masks([low_res_masks], [raw_video.shape[-3:-1]], binarize=False)[
            0
        ]
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-13.5762, -13.5762, -13.7167], [-13.0870, -13.0870, -13.5405], [-12.2173, -12.2173, -13.2273]]
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
                    [[[[-13.5762, -13.5762], [-13.0870, -13.0870]]]],
                    [[[[-19.1203, -19.1203], [-19.5488, -19.5488]]]],
                    [[[[-19.9951, -19.9951], [-20.5353, -20.5353]]]],
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
                    [[[[-13.5762, -13.5762], [-13.0870, -13.0870]]]],
                    [[[[-19.1203, -19.1203], [-19.5488, -19.5488]]]],
                    [[[[-19.9951, -19.9951], [-20.5353, -20.5353]]]],
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
        self.assertEqual(low_res_masks.shape, (1, 1, 288, 288))
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-11.9889, -11.9889, -12.2238], [-11.6383, -11.6383, -12.0873], [-11.0150, -11.0150, -11.8446]]
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
                    [[[[-11.9889, -11.9889], [-11.6383, -11.6383]]]],
                    [[[[-20.4502, -20.4502], [-20.6929, -20.6929]]]],
                    [[[[-22.0344, -22.0344], [-22.4522, -22.4522]]]],
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
        self.assertEqual(low_res_masks.shape, (1, 1, 288, 288))
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-17.2589, -17.2589, -17.5130], [-17.2777, -17.2777, -17.9154], [-17.3111, -17.3111, -18.6309]]
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
                    [[[[-17.2589, -17.2589], [-17.2777, -17.2777]]]],
                    [[[[-17.8107, -17.8107], [-18.1581, -18.1581]]]],
                    [[[[-17.9432, -17.9432], [-18.4637, -18.4637]]]],
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
        self.assertEqual(low_res_masks.shape, (1, 1, 288, 288))
        self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[0, 0, :3, :3],
            torch.tensor(
                [[-14.0206, -14.0206, -14.1225], [-14.0568, -14.0568, -14.4570], [-14.1212, -14.1212, -15.0516]]
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
                    [[[[-14.0206, -14.0206], [-14.0568, -14.0568]]]],
                    [[[[-16.8155, -16.8155], [-17.2954, -17.2954]]]],
                    [[[[-16.2909, -16.2909], [-16.8887, -16.8887]]]],
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
        self.assertEqual(low_res_masks.shape, (2, 1, 288, 288))
        self.assertEqual(video_res_masks.shape, (2, 1, raw_video.shape[-3], raw_video.shape[-2]))
        torch.testing.assert_close(
            video_res_masks[:, 0, :2, :2],  # first object
            torch.tensor(
                [[[-12.8567, -12.8567], [-13.0618, -13.0618]], [[-12.1054, -12.1054], [-11.6056, -11.6056]]]
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
                    [[[[-12.8567, -12.8567], [-13.0618, -13.0618]]], [[[-12.1054, -12.1054], [-11.6056, -11.6056]]]],
                    [[[[-22.5194, -22.5194], [-22.7973, -22.7973]]], [[[-20.6199, -20.6199], [-21.0607, -21.0607]]]],
                    [[[[-25.0871, -25.0871], [-25.6355, -25.6355]]], [[[-19.9508, -19.9508], [-20.4212, -20.4212]]]],
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_inference_mask_generation_video_batched_bb(self):
        raw_video = prepare_video()
        inference_session = self.processor.init_video_session(video=raw_video, inference_device=torch_device)
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_ids = [2, 3]  # give a unique id to each object we interact with (it can be any integers)

        self.processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=ann_obj_ids,
            input_boxes=[[[300, 0, 500, 400], [400, 0, 600, 400]]],
        )

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
                    [[[[-17.2589, -17.2589], [-17.2777, -17.2777]]], [[[-8.5523, -8.5523], [-8.5103, -8.5103]]]],
                    [[[[-17.8107, -17.8107], [-18.1581, -18.1581]]], [[[-9.1150, -9.1150], [-9.2327, -9.2327]]]],
                    [[[[-17.9432, -17.9432], [-18.4637, -18.4637]]], [[[-10.9026, -10.9026], [-11.0184, -11.0184]]]],
                ]
            ).to(torch_device),
            atol=1e-4,
            rtol=1e-3,
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
        self.assertEqual(low_res_masks.shape, (1, 1, 288, 288))
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
                    [[[[-21.3700, -21.3700], [-21.7191, -21.7191]]]],
                    [[[[-22.2242, -22.2242], [-22.7148, -22.7148]]]],
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
                    [[[[-11.9889, -11.9889], [-11.6383, -11.6383]]]],
                    [[[[-20.4502, -20.4502], [-20.6929, -20.6929]]]],
                    [[[[-22.0344, -22.0344], [-22.4522, -22.4522]]]],
                ]
            ).to(torch_device),
            atol=1e-2,
            rtol=1e-2,
        )

    def test_inference_with_different_dtypes(self):
        """Test that inference works correctly for float32, bfloat16, and float16 dtypes."""
        raw_video = prepare_video()
        dtypes_to_test = [
            (torch.float32, None),  # float32 is always available
            (torch.bfloat16, is_torch_bf16_available_on_device),
            (torch.float16, is_torch_fp16_available_on_device),
        ]

        for dtype, availability_check in dtypes_to_test:
            with self.subTest(dtype=dtype):
                # Skip if dtype is not available on device
                if availability_check is not None and not availability_check(torch_device):
                    self.skipTest(f"{dtype} not supported on {torch_device}")

                # Load model with specific dtype
                video_model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", torch_dtype=dtype).to(
                    torch_device
                )
                video_model.eval()

                # Initialize inference session
                inference_session = self.processor.init_video_session(
                    video=raw_video, inference_device=torch_device, dtype=dtype
                )
                ann_frame_idx = 0
                ann_obj_id = 1

                # Add inputs
                self.processor.add_inputs_to_inference_session(
                    inference_session=inference_session,
                    frame_idx=ann_frame_idx,
                    obj_ids=ann_obj_id,
                    input_points=[[[[210, 350]]]],
                    input_labels=[[[1]]],
                )

                # Run inference on first frame
                outputs = video_model(inference_session=inference_session, frame_idx=ann_frame_idx)
                low_res_masks = outputs.pred_masks

                # Verify output shape and dtype
                self.assertEqual(low_res_masks.shape, (1, 1, 288, 288))
                self.assertEqual(low_res_masks.dtype, dtype)

                # Post-process masks
                video_res_masks = self.processor.post_process_masks(
                    [low_res_masks], [raw_video.shape[-3:-1]], binarize=False
                )[0]
                self.assertEqual(video_res_masks.shape, (1, 1, raw_video.shape[-3], raw_video.shape[-2]))

                # Test propagation across multiple frames to test memory handling
                frames = []
                max_frame_num_to_track = 2
                for sam2_video_output in video_model.propagate_in_video_iterator(
                    inference_session=inference_session,
                    start_frame_idx=ann_frame_idx,
                    max_frame_num_to_track=max_frame_num_to_track,
                ):
                    video_res_masks = self.processor.post_process_masks(
                        [sam2_video_output.pred_masks], [raw_video.shape[-3:-1]], binarize=False
                    )[0]
                    frames.append(video_res_masks)
                    # Verify dtype is maintained during propagation
                    self.assertEqual(sam2_video_output.pred_masks.dtype, dtype)

                frames = torch.stack(frames, dim=0)
                # Verify we got the expected number of frames (initial frame + max_frame_num_to_track)
                self.assertEqual(
                    frames.shape, (max_frame_num_to_track + 1, 1, 1, raw_video.shape[-3], raw_video.shape[-2])
                )
                # Verify dtype is maintained in stacked frames
                self.assertEqual(frames.dtype, dtype)
