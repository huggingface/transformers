# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

import gc
import logging

import numpy as np

import torch
from PIL import Image

from .act_ckpt_utils import clone_output_wrapper

from .box_ops import box_xywh_to_cxcywh, box_xyxy_to_xywh

from .data_misc import (
    BatchedDatapoint,
    BatchedPointer,
    convert_my_tensors,
    FindStage,
    recursive_to,
)

from .geometry_encoders import Prompt
from .model_misc import NestedTensor
from .sam3_image import Sam3Image


def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size  # the original video size
    return img, video_height, video_width


def load_image_as_single_frame(
    image_path,
    image_size,
    offload_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
):
    """Load an image as a single-frame video."""
    images, image_height, image_width = _load_img_as_tensor(image_path, image_size)
    images = images.unsqueeze(0).half()

    img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]
    if not offload_to_cpu:
        images = images.cuda()
        img_mean = img_mean.cuda()
        img_std = img_std.cuda()
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, image_height, image_width


class Sam3ImageInteractiveDemo(Sam3Image):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2

    def __init__(
        self,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        compile_model=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.compile_model = compile_model

        # Debug added for owlvit
        self.use_aot_mem = False
        self.use_obj_mem_bank = False

    @torch.inference_mode()
    def init_state(
        self,
        resource_path,
        offload_to_cpu=False,
    ):
        """Initialize an inference state from `resource_path` (an image or a video)."""

        images, orig_height, orig_width = load_image_as_single_frame(
            image_path=resource_path,
            image_size=self.image_size,
            offload_to_cpu=offload_to_cpu,
        )
        inference_state = {}
        inference_state["image_size"] = self.image_size
        inference_state["num_frames"] = len(images)
        inference_state["device"] = torch.device("cuda")
        # the original video height and width, used for resizing final output scores
        inference_state["orig_height"] = orig_height
        inference_state["orig_width"] = orig_width
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # inputs on each frame
        self._construct_initial_input_batch(inference_state, images)
        return inference_state

    def _construct_initial_input_batch(self, inference_state, images):
        """Construct an initial `BatchedDatapoint` instance as input."""
        # 1) img_batch
        num_frames = len(images)
        device = inference_state["device"]
        img_batch = NestedTensor(tensors=images, mask=None)

        # 2) find_text_batch
        # "<text placeholder>" will be replaced by the actual text prompt when adding prompts
        find_text_batch = ["<text placeholder>", "visual", "geometric"]

        # 3) find_inputs
        input_box_embedding_dim = 258  # historical default
        input_points_embedding_dim = 257  # historical default
        dummy_ptrs = BatchedPointer(
            stage_ids=[], query_ids=[], object_ids=[], ptr_mask=[], ptr_types=[]
        )
        stages = [
            FindStage(
                img_ids=[stage_id],
                text_ids=[0],
                input_boxes=[torch.zeros(input_box_embedding_dim)],
                input_boxes_before_embed=[torch.zeros(4)],
                input_boxes_mask=[1],
                input_boxes_label=[0],
                input_points=[torch.empty(0, input_points_embedding_dim)],
                input_points_before_embed=[torch.empty(0, 3)],
                input_points_mask=[torch.empty(0)],
                ptrs=dummy_ptrs,
                ptrs_seg=dummy_ptrs,
                object_ids=[],
            )
            for stage_id in range(num_frames)
        ]
        for i in range(len(stages)):
            stages[i] = convert_my_tensors(stages[i])

        # construct the final `BatchedDatapoint` and cast to GPU
        input_batch = BatchedDatapoint(
            img_batch=img_batch,
            find_text_batch=find_text_batch,
            find_inputs=stages,
            find_targets=[None] * num_frames,
            get_queries=None,
            find_metadatas=[None] * num_frames,
        )
        input_batch = recursive_to(input_batch, device, non_blocking=True)
        inference_state["input_batch"] = input_batch

        # construct the placeholder interactive prompts and tracking queries
        bs = 1
        inference_state["constants"]["empty_geometric_prompt"] = Prompt(
            box_embeddings=torch.zeros(0, bs, 4, device=device),
            box_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
            box_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
            point_embeddings=torch.zeros(0, bs, 2, device=device),
            point_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
            point_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
        )

        # constructing an output list in inference state (we start with an empty list)
        inference_state["previous_stages_out"] = [None] * num_frames
        inference_state["text_prompt"] = None
        inference_state["per_frame_raw_point_input"] = [None] * num_frames
        inference_state["per_frame_raw_box_input"] = [None] * num_frames
        inference_state["per_frame_visual_prompt"] = [None] * num_frames
        inference_state["per_frame_geometric_prompt"] = [None] * num_frames
        inference_state["per_frame_cur_step"] = [0] * num_frames
        if self.use_aot_mem:
            inference_state["aot_mem_per_frame"] = {"spatial": {}, "pointer": {}}
        if self.use_obj_mem_bank:
            inference_state["obj_mem_per_frame"] = {
                "roi": {},
                "roi_zoomed_out": {},
                "global": {},
            }

        # placeholders for cached outputs
        # (note: currently, a single visual prompt embedding is shared for all frames)
        inference_state["backbone_out"] = None
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None

    @torch.inference_mode()
    def add_prompt(
        self,
        inference_state,
        frame_idx=0,
        text_str=None,
        clear_old_points=True,
        points=None,
        point_labels=None,
        boxes_xywh=None,
        box_labels=None,
        clear_old_boxes=True,
        output_prob_thresh=0.5,
        instance_prompt=False,
    ):
        """
        Add text, point or box prompts on a single frame. This method returns the inference
        outputs only on the prompted frame.

        Note that text prompts are NOT associated with a particular frame (i.e. they apply
        to all frames). However, we only run inference on the frame specified in `frame_idx`.
        """
        device = inference_state["device"]
        num_frames = inference_state["num_frames"]
        assert (
            text_str is not None or points is not None or boxes_xywh is not None
        ), "at least one type of prompt (text, points, boxes) must be provided"
        assert (
            0 <= frame_idx < num_frames
        ), f"{frame_idx=} is out of range for a total of {num_frames} frames"

        # 1) add text prompt
        if text_str is not None:
            # currently we do not allow simultaneously adding text prompt and visual
            # prompt both as initial prompt (since visual prompt uses the text "visual")
            if any(p is not None for p in inference_state["per_frame_visual_prompt"]):
                raise RuntimeError(
                    "Text and visual prompts (box as an initial prompt) cannot be used together. "
                    "Please reset the session."
                )

            inference_state["text_prompt"] = text_str
            # add the text prompt into the input batch (to be applied to *all* frames)
            inference_state["input_batch"].find_text_batch[0] = text_str
            for t in range(inference_state["num_frames"]):
                text_id = self.TEXT_ID_FOR_TEXT
                inference_state["input_batch"].find_inputs[t].text_ids[...] = text_id

        # 2) add geometric prompt (points or boxes)
        # start with an empty geometric_prompt (we later add previous point and box prompts
        # from "per_frame_raw_point_input" and "per_frame_raw_box_input" below)
        geometric_prompt = inference_state["constants"][
            "empty_geometric_prompt"
        ].clone()

        if points is not None and boxes_xywh is not None:
            raise RuntimeError(
                "Cannot add both point and box prompts at the same time. "
            )

        if points is not None and not instance_prompt:
            raise RuntimeError(
                "Point prompts are only supported for instance tracking. "
            )

        if instance_prompt and (text_str is not None or boxes_xywh is not None):
            raise RuntimeError(
                "Text and box prompts are not supported for instance tracking. "
            )

        new_visual_prompt = None

        # 2.1) handle point prompt
        assert (points is not None) == (point_labels is not None)
        if points is not None:
            points = torch.as_tensor(points, dtype=torch.float32)
            point_labels = torch.as_tensor(point_labels, dtype=torch.long)
            assert points.dim() == 2
            assert points.size(0) > 0 and points.size(-1) == 2
            assert point_labels.dim() == 1 and point_labels.size(0) == points.size(0)
            assert torch.all(points >= 0).item() and torch.all(points <= 1).item()
            # append previous points under `clear_old_points=False`
            prev_point_input = inference_state["per_frame_raw_point_input"][frame_idx]
            if prev_point_input is not None and not clear_old_points:
                prev_points, prev_point_labels = prev_point_input
                points = torch.cat([prev_points, points], dim=0)
                point_labels = torch.cat([prev_point_labels, point_labels], dim=0)
            new_point_input = points, point_labels
            inference_state["per_frame_raw_point_input"][frame_idx] = new_point_input
            # add a batch dimensions (note that it's sequence first)
            points = points.unsqueeze(1).to(device)
            point_labels = point_labels.unsqueeze(1).to(device)
            geometric_prompt.append_points(points=points, labels=point_labels)
            new_visual_prompt = None

            for t in range(inference_state["num_frames"]):
                inference_state["input_batch"].find_inputs[t].text_ids[
                    ...
                ] = self.TEXT_ID_FOR_GEOMETRIC

        # 2.2) handle box prompt
        assert (boxes_xywh is not None) == (box_labels is not None)
        if boxes_xywh is not None:
            boxes_xywh = torch.as_tensor(boxes_xywh, dtype=torch.float32)
            box_labels = torch.as_tensor(box_labels, dtype=torch.long)
            # input boxes are expected to be [xmin, ymin, width, height] format
            # in normalized coordinates of range 0~1, similar to FA
            assert boxes_xywh.dim() == 2
            assert boxes_xywh.size(0) > 0 and boxes_xywh.size(-1) == 4
            assert box_labels.dim() == 1 and box_labels.size(0) == boxes_xywh.size(0)
            boxes_cxcywh = box_xywh_to_cxcywh(boxes_xywh)
            assert (boxes_xywh >= 0).all().item() and (boxes_xywh <= 1).all().item()
            assert (boxes_cxcywh >= 0).all().item() and (boxes_cxcywh <= 1).all().item()
            # append previous boxes under `clear_old_boxes=False`
            prev_box_input = inference_state["per_frame_raw_box_input"][frame_idx]
            if prev_box_input is not None and not clear_old_boxes:
                prev_boxes_cxcywh, prev_box_labels = prev_box_input
                boxes_cxcywh = torch.cat([prev_boxes_cxcywh, boxes_cxcywh], dim=0)
                box_labels = torch.cat([prev_box_labels, box_labels], dim=0)
            new_box_input = boxes_cxcywh, box_labels
            inference_state["per_frame_raw_box_input"][frame_idx] = new_box_input

            # handle the case of visual prompt (also added as an input box from the UI)
            boxes_cxcywh, box_labels, new_visual_prompt = self._get_visual_prompt(
                inference_state, frame_idx, boxes_cxcywh, box_labels
            )
            # add a batch dimensions (note that it's sequence first)
            boxes_cxcywh = boxes_cxcywh.unsqueeze(1).to(device)
            box_labels = box_labels.unsqueeze(1).to(device)
            geometric_prompt.append_boxes(boxes=boxes_cxcywh, labels=box_labels)

        inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt

        # 3) run inference on this frame
        inference_state["backbone_out"] = self._init_backbone_out(inference_state)
        if new_visual_prompt is not None:
            # currently we do not allow simultaneously adding text prompt and visual
            # prompt both as initial prompt (since visual prompt uses the text "visual")
            if inference_state["text_prompt"] is not None:
                raise RuntimeError(
                    "Text and visual prompts (box as an initial prompt) cannot be used together. "
                    "Please reset the session."
                )

            # add the visual prompt into the input batch and encode it (currently the added
            # visual prompt is applied to *all* frames, i.e. not just this prompted frame)
            for t in range(inference_state["num_frames"]):
                text_id = self.TEXT_ID_FOR_VISUAL
                inference_state["input_batch"].find_inputs[t].text_ids[...] = text_id
            # currently visual prompt is encoded the same way (`_encode_prompt`) as geometric prompt
            visual_prompt_embed, visual_prompt_mask, _backbone_out = (
                self._encode_prompt(
                    backbone_out=inference_state["backbone_out"],
                    find_input=inference_state["input_batch"].find_inputs[frame_idx],
                    geometric_prompt=new_visual_prompt,
                    encode_text=False,
                )
            )
            inference_state["visual_prompt_embed"] = visual_prompt_embed
            inference_state["visual_prompt_mask"] = visual_prompt_mask

        reverse = False  # TODO for now, we always track forward when adding prompts
        out = self._run_single_frame_inference(
            inference_state,
            frame_idx,
            reverse,
            is_instance_processing=instance_prompt,
        )
        return self._postprocess_output(
            inference_state, out, output_prob_thresh, pop=False
        )

    def _get_visual_prompt(self, inference_state, frame_idx, boxes_cxcywh, box_labels):
        """
        Handle the case of visual prompt. Currently, in the inference API we do not
        explicitly distinguish between initial box as visual prompt vs subsequent boxes
        or boxes after inference for refinement.
        """
        # If the frame hasn't had any inference results before (prompting or propagation),
        # we treat the first added box prompt as a visual prompt; otherwise, we treat
        # the first box just as a refinement prompt.
        is_new_visual_prompt = (
            inference_state["per_frame_visual_prompt"][frame_idx] is None
            and inference_state["previous_stages_out"][frame_idx] is None
        )
        if is_new_visual_prompt:
            # take the first box prompt as a visual prompt
            device = inference_state["device"]
            new_visual_prompt = Prompt(
                box_embeddings=boxes_cxcywh[None, 0:1, :].to(device),  # (seq, bs, 4)
                box_mask=None,
                box_labels=box_labels[None, 0:1].to(device),  # (seq, bs)
                point_embeddings=None,
                point_mask=None,
                point_labels=None,
            )
            inference_state["per_frame_visual_prompt"][frame_idx] = new_visual_prompt
        else:
            new_visual_prompt = None

        # `boxes_cxcywh` and `box_labels` contains all the raw box inputs added so far
        # strip any visual prompt from the input boxes (for geometric prompt encoding)
        if inference_state["per_frame_visual_prompt"][frame_idx] is not None:
            boxes_cxcywh = boxes_cxcywh[1:]
            box_labels = box_labels[1:]

        return boxes_cxcywh, box_labels, new_visual_prompt

    def _init_backbone_out(self, inference_state):
        """
        Initialize a backbone_out dictionary and extract the text features.

        Note that the visual features of each frame are not extracted here. They will be
        extracted on the fly when running inference on each frame.
        """
        input = inference_state["input_batch"]
        device = self.device
        backbone_out = {"img_batch_all_stages": input.img_batch}
        text_outputs = self.backbone.forward_text(input.find_text_batch, device=device)
        backbone_out.update(text_outputs)
        return backbone_out

    def _run_single_frame_inference(
        self, inference_state, frame_idx, reverse, is_instance_processing=False
    ):
        """
        Perform inference on a single frame and get its inference results. This would
        also update `inference_state`.
        """
        input = inference_state["input_batch"]
        find_input = input.find_inputs[frame_idx]
        find_target = None
        num_frames = inference_state["num_frames"]
        is_video_batch = num_frames > 1

        backbone_out = inference_state["backbone_out"]
        geometric_prompt = inference_state["per_frame_geometric_prompt"][frame_idx]
        if geometric_prompt is None:
            geometric_prompt = inference_state["constants"]["empty_geometric_prompt"]
        previous_stages_out = inference_state["previous_stages_out"]
        prev_encoder_out = None
        if previous_stages_out[frame_idx] is not None:
            prev_encoder_out = previous_stages_out[frame_idx].get("prev_encoder_out")
        cur_step = inference_state["per_frame_cur_step"][frame_idx]

        if self.use_aot_mem:
            aot_mem_per_frame = inference_state["aot_mem_per_frame"]
            aot_mem_id_bank = self._aot_mem_get_rand_permuted_id_bank(is_video_batch)
        else:
            aot_mem_per_frame, aot_mem_id_bank = None, None
        obj_mem_per_frame = inference_state.get("obj_mem_per_frame", None)

        prev_mask_pred = None
        if (
            inference_state["previous_stages_out"][frame_idx]
            # and self.use_prev_mask
            and is_instance_processing
        ):
            prev_mask_pred = self._get_best_mask(
                inference_state["previous_stages_out"][frame_idx]
            )

        out, _ = self.forward_video_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=find_target,
            frame_idx=frame_idx,
            # num_frames=num_frames,
            previous_stages_out=previous_stages_out,
            geometric_prompt=geometric_prompt.clone(),
            # run_encoder=self.interactivity_in_encoder or cur_step == 0,
            prev_encoder_out=prev_encoder_out,
            visual_prompt=inference_state["visual_prompt_embed"],
            visual_prompt_mask=inference_state["visual_prompt_mask"],
            is_instance_prompt=is_instance_processing,
            track_in_reverse=reverse,
            prev_mask_pred=prev_mask_pred,
        )
        inference_state["previous_stages_out"][frame_idx] = out
        inference_state["per_frame_cur_step"][frame_idx] = cur_step + 1

        return out

    def _postprocess_output(
        self,
        inference_state,
        out,
        output_prob_thresh,
        pop=True,
    ):
        """Post-process the single-frame output into the desired numpy result format."""
        prompt_idx = 0
        if pop:
            out_scores = out.pop("pred_logits")[prompt_idx].squeeze(-1)
            out_boxes_xyxy = out.pop("pred_boxes_xyxy")[prompt_idx]
            # out_obj_ids = out.pop("pred_object_ids")[prompt_idx]
            out_masks = out.pop("pred_masks")[prompt_idx]

            # remove a few unused keys (to reduce GPU memory usage)
            unused_output_keys = [
                "pred_boxes",
                "pred_is_valid",
                "pred_old_obj_ids",
                "semantic_seg",
                "presence_logit",
            ]
            for k in unused_output_keys:
                out.pop(k, None)
        else:
            out_scores = out["pred_logits"][prompt_idx].squeeze(-1)
            out_boxes_xyxy = out["pred_boxes_xyxy"][prompt_idx]
            # out_obj_ids = out["pred_object_ids"][prompt_idx]
            out_masks = out["pred_masks"][prompt_idx]

        # only take the entries above the score threshold
        out_probs = out_scores.sigmoid()  # output in probabilities in 0~1
        # keep = out_obj_ids >= 0
        if output_prob_thresh is not None:
            # keep = torch.logical_and(out_probs > output_prob_thresh, keep)
            keep = out_probs > output_prob_thresh
        out_probs = out_probs[keep]
        out_boxes_xyxy = out_boxes_xyxy[keep]
        # out_obj_ids = out_obj_ids[keep]
        out_masks = out_masks[keep]

        out_boxes_xywh = box_xyxy_to_xywh(out_boxes_xyxy)  # output in XYWH box format
        num_out_obj = out_masks.size(0)
        if num_out_obj > 0:
            out_masks_orig_size = torch.nn.functional.interpolate(
                out_masks.unsqueeze(0),
                size=(inference_state["orig_height"], inference_state["orig_width"]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            out_binary_masks = out_masks_orig_size > 0.0
        else:
            # in case there is no object, `torch.nn.functional.interpolate` would raise
            # an error on an empty tensor, so we treat it specially here
            out_binary_masks = torch.zeros(
                0,
                inference_state["orig_height"],
                inference_state["orig_width"],
                dtype=torch.bool,
            )

        # We directly convert the outputs to CPU numpy format so that it is easy for
        # the server to send them across processes and construct the final response.
        frame_outputs = {
            "out_probs": out_probs.float().cpu().numpy(),
            "out_boxes_xywh": out_boxes_xywh.float().cpu().numpy(),
            # "out_obj_ids": out_obj_ids.cpu().numpy(),
            "out_binary_masks": out_binary_masks.cpu().numpy(),
        }

        return frame_outputs

    @torch.inference_mode()
    def reset_state(self, inference_state):
        """Revert `inference_state` to what it was right after initialization."""
        inference_state["input_batch"].find_text_batch[0] = "<text placeholder>"
        inference_state["text_prompt"] = None
        for t in range(inference_state["num_frames"]):
            inference_state["input_batch"].find_inputs[t].text_ids[...] = 0
            # constructing an output list in inference state (we start with an empty list)
            inference_state["previous_stages_out"][t] = None
            inference_state["per_frame_raw_point_input"][t] = None
            inference_state["per_frame_raw_box_input"][t] = None
            inference_state["per_frame_visual_prompt"][t] = None
            inference_state["per_frame_geometric_prompt"][t] = None
            inference_state["per_frame_cur_step"][t] = 0

        inference_state["backbone_out"] = None
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None
        if self.use_aot_mem:
            inference_state["aot_mem_per_frame"]["spatial"].clear()
            inference_state["aot_mem_per_frame"]["pointer"].clear()
        if self.use_obj_mem_bank:
            inference_state["obj_mem_per_frame"]["roi"].clear()
            inference_state["obj_mem_per_frame"]["roi_zoomed_out"].clear()
            inference_state["obj_mem_per_frame"]["global"].clear()
        gc.collect()

    def _compile_model(self):
        """Compile the SAM model with torch.compile for speedup."""
        is_compiled = getattr(self, "_model_is_compiled", False)
        if is_compiled or not self.compile_model:
            return

        import torch._dynamo

        # a larger cache size to hold varying number of shapes for torch.compile
        # see https://github.com/pytorch/pytorch/blob/v2.5.1/torch/_dynamo/config.py#L42-L49
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True

        self.backbone.vision_backbone.forward = clone_output_wrapper(
            torch.compile(
                self.backbone.vision_backbone.forward,
                fullgraph=True,
                mode="max-autotune",
            )
        )
        self.transformer.encoder.forward = clone_output_wrapper(
            torch.compile(
                self.transformer.encoder.forward,
                fullgraph=True,
                mode="max-autotune",
            )
        )
        self.transformer.decoder.forward = clone_output_wrapper(
            torch.compile(
                self.transformer.decoder.forward,
                fullgraph=True,
                mode="max-autotune",
                dynamic=True,  # the decoder uses dynamic shapes
            )
        )
        self._model_is_compiled = True

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def warm_up_compilation(self):
        """
        Warm up the model by running a dummy inference to compile the model. This is
        useful to avoid the compilation overhead in the first inference call.
        """
        if not self.compile_model:
            return

        if self.device.type != "cuda":
            raise RuntimeError(
                f"The model must be on CUDA for warm-up compilation, got {self.device=}."
            )

        # Get a random video
        orig_tracking_score_thresh = self.tracking_score_thresh
        inference_state = self.init_state(resource_path="<load-dummy-video-30>")
        # use different tracking score thresholds for each round to simulate different number of output objects
        tracking_score_thresh_list = [0.0, -3.0, -3.5, -4.0]
        num_rounds = len(tracking_score_thresh_list)
        for i, thresh in enumerate(tracking_score_thresh_list):
            self.tracking_score_thresh = thresh
            # start at different locations for each round
            start_frame_idx = (inference_state["num_frames"] * i) // num_rounds
            logging.warning(f"{i+1}/{num_rounds} warming up model compilation")
            self.add_prompt(inference_state, frame_idx=start_frame_idx, text_str="cat")
            logging.warning(
                f"{i+1}/{num_rounds} warming up model compilation with propagation -- this might take a while"
            )
            for _ in self.propagate_in_video(
                inference_state, start_frame_idx, reverse=False
            ):
                pass
            for _ in self.propagate_in_video(
                inference_state, start_frame_idx, reverse=True
            ):
                pass
            self.reset_state(inference_state)
            logging.warning(
                f"{i+1}/{num_rounds} warming up model compilation -- completed round {i+1} out of {num_rounds}"
            )

        self.tracking_score_thresh = orig_tracking_score_thresh
        logging.warning("Warm-up compilation completed.")
