# Copyright (c) Meta, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Optional

import torch

from .act_ckpt_utils import activation_ckpt_wrapper

from .box_ops import box_cxcywh_to_xyxy

from .geometry_encoders import Prompt
from .model_misc import inverse_sigmoid, NestedTensor


def _update_out(out, out_name, out_value, auxiliary=True):
    out[out_name] = out_value[-1] if auxiliary else out_value
    if auxiliary:
        if "aux_outputs" not in out:
            out["aux_outputs"] = [{} for _ in range(len(out_value) - 1)]
        assert len(out["aux_outputs"]) == len(out_value) - 1
        for aux_output, aux_value in zip(out["aux_outputs"], out_value[:-1]):
            aux_output[out_name] = aux_value


class Sam3Image(torch.nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head=None,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=None,
        use_instance_query: bool = True,
        multimask_output: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.geometry_encoder = input_geometry_encoder
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.segmentation_head = segmentation_head

        self.o2m_mask_predict = o2m_mask_predict

        self.dot_prod_scoring = dot_prod_scoring

        # verify the number of queries for O2O and O2M
        num_o2o_static = self.transformer.decoder.num_queries
        num_o2m_static = self.transformer.decoder.num_o2m_queries
        assert num_o2m_static == (num_o2o_static if self.transformer.decoder.dac else 0)

        self.use_instance_query = use_instance_query
        self.multimask_output = multimask_output

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        # clear cached _device in case the model is moved to a different device
        self._device = None
        return super().to(*args, **kwargs)

    def _get_img_feats(self, backbone_out, img_ids):
        """Retrieve correct image features from backbone output."""
        if "backbone_fpn" in backbone_out:
            if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
                # this is for the video case. We only have a partial forward of all features
                img_ids = backbone_out["id_mapping"][img_ids]
                # If this assert fails, it likely means we're requesting different img_ids (perhaps a different frame?)
                # We currently don't expect this to happen. We could technically trigger a recompute here,
                # but likely at the cost of a cpu<->gpu sync point, which would deteriorate perf
                torch._assert_async((img_ids >= 0).all())

            vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels :]
            vis_pos_enc = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
            vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]  # (H, W) shapes
            # index and flatten visual features NxCxHxW => HWxNxC (batch-first => seq-first)
            img_feats = [
                x.tensors[img_ids].flatten(2).permute(2, 0, 1) for x in vis_feats
            ]
            img_masks = [
                None if x.mask is None else x.mask[img_ids].flatten(1)
                for x in vis_feats
            ]
            img_pos_embeds = [
                x[img_ids].flatten(2).permute(2, 0, 1) for x in vis_pos_enc
            ]
            return backbone_out, img_feats, img_masks, img_pos_embeds, vis_feat_sizes

        # Image features not available in backbone output, so we compute them on the fly
        # This case likely occurs for video. In that case, we want to forward only the current frame
        img_batch = backbone_out["img_batch_all_stages"]
        if img_ids.numel() > 1:
            # Only forward backbone on unique image ids to avoid repetitive computation
            unique_ids, _ = torch.unique(img_ids, return_inverse=True)
        else:
            unique_ids, _ = img_ids, slice(None)
        # Compute the image features on those unique image ids
        # note: we allow using a list (or other indexable types) of tensors as img_batch.tensors
        # (e.g. for async frame loading in demo). In this case we index img_batch.tensors directly
        if isinstance(img_batch.tensors, torch.Tensor):
            image = img_batch.tensors[unique_ids]
        elif unique_ids.numel() == 1:
            image = img_batch.tensors[unique_ids.item()].unsqueeze(0)
        else:
            image = torch.stack([img_batch.tensors[i] for i in unique_ids.tolist()])
        # `img_batch` might be fp16 and offloaded to CPU
        image = image.to(dtype=torch.float32, device=self.device)
        image_mask = img_batch.mask[unique_ids] if img_batch.mask is not None else None
        image_tensors = NestedTensor(tensors=image, mask=image_mask)
        # Next time we call this function, we want to remember which indices we computed
        id_mapping = torch.full(
            (len(img_batch.tensors),), -1, dtype=torch.long, device=self.device
        )
        id_mapping[unique_ids] = torch.arange(len(unique_ids), device=self.device)
        backbone_out = {
            **backbone_out,
            **self.backbone.forward_image(image_tensors),
            "id_mapping": id_mapping,
        }
        assert "backbone_fpn" in backbone_out
        return self._get_img_feats(backbone_out, img_ids=img_ids)

    def _get_feat_tuple(self, backbone_out, find_input):
        img_ids, img_feat_inds = find_input.img_ids, slice(None)
        return self._get_img_feats(backbone_out, img_ids), img_feat_inds

    def _encode_prompt(
        self,
        backbone_out,
        find_input,
        geometric_prompt,
        visual_prompt_embed=None,
        visual_prompt_mask=None,
        encode_text=True,
        prev_mask_pred=None,
    ):
        # index text features (note that regardless of early or late fusion, the batch size of
        # `txt_feats` is always the number of *prompts* in the encoder)
        txt_ids = find_input.text_ids
        txt_feats = backbone_out["language_features"][:, txt_ids]
        txt_masks = backbone_out["language_mask"][txt_ids]

        feat_tuple, _ = self._get_feat_tuple(backbone_out, find_input)
        backbone_out, img_feats, img_masks, img_pos_embeds, vis_feat_sizes = feat_tuple

        if prev_mask_pred is not None:
            # TODO: Support Multi-scale? for now, mutli-scale will break other things (like decoder boxRPB), so it won't go silently.
            img_feats = [img_feats[-1] + prev_mask_pred]
        # Encode geometry
        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )
        if visual_prompt_embed is None:
            visual_prompt_embed = torch.zeros(
                (0, *geo_feats.shape[1:]), device=geo_feats.device
            )
            visual_prompt_mask = torch.zeros(
                (*geo_masks.shape[:-1], 0),
                device=geo_masks.device,
                dtype=geo_masks.dtype,
            )
        if encode_text:
            prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([txt_masks, geo_masks, visual_prompt_mask], dim=1)
        else:
            prompt = torch.cat([geo_feats, visual_prompt_embed], dim=0)
            prompt_mask = torch.cat([geo_masks, visual_prompt_mask], dim=1)
        return prompt, prompt_mask, backbone_out

    def _run_encoder(
        self,
        backbone_out,
        find_input,
        prompt,
        prompt_mask,
        prev_mask_pred=None,
        encoder_extra_kwargs: Optional[Dict] = None,
    ):
        feat_tuple, img_feat_inds = self._get_feat_tuple(backbone_out, find_input)
        backbone_out, img_feats, img_masks, img_pos_embeds, vis_feat_sizes = feat_tuple
        if prev_mask_pred is not None:
            # TODO: Support Multi-scale? for now, mutli-scale will break other things (like decoder boxRPB), so it won't go silently.
            img_feats = [img_feats[-1] + prev_mask_pred]
        # Run the encoder
        prompt_pos_embed = torch.zeros_like(prompt)
        # make a copy of the image feature lists since the encoder may modify these lists in-place
        memory = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=img_masks.copy(),
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
            encoder_extra_kwargs=encoder_extra_kwargs,
        )
        encoder_out = {
            # encoded image features
            "encoder_hidden_states": memory["memory"],
            "pos_embed": memory["pos_embed"],
            "padding_mask": memory["padding_mask"],
            "level_start_index": memory["level_start_index"],
            "spatial_shapes": memory["spatial_shapes"],
            "valid_ratios": memory["valid_ratios"],
            "vis_feat_sizes": vis_feat_sizes,
            "img_feat_inds": img_feat_inds,
            # encoded text features (or other prompts)
            "prompt_before_enc": prompt,
            "prompt_after_enc": memory.get("memory_text", prompt),
            "prompt_mask": prompt_mask,
        }
        return backbone_out, encoder_out, feat_tuple

    def _update_scores_and_boxes(
        self, out, hs, reference_boxes, prompt, prompt_mask, apply_dac=None
    ):
        apply_dac = apply_dac if apply_dac is not None else self.transformer.decoder.dac
        num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
        num_o2m = hs.size(2) - num_o2o
        assert num_o2m == (num_o2o if apply_dac else 0)
        out["queries"] = hs[-1][:, :num_o2o]  # remove o2m queries if there are any
        # score prediction
        # if self.use_dot_prod_scoring:
        outputs_class = self.dot_prod_scoring(hs, prompt, prompt_mask)
        # else:
        #     outputs_class = self.class_embed(hs)
        # box prediction
        anchor_box_offsets = self.transformer.decoder.bbox_embed(hs)
        reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid()
        outputs_boxes_xyxy = box_cxcywh_to_xyxy(outputs_coord)

        _update_out(out, "pred_logits", outputs_class[:, :, :num_o2o])
        _update_out(out, "pred_boxes", outputs_coord[:, :, :num_o2o])
        _update_out(out, "pred_boxes_xyxy", outputs_boxes_xyxy[:, :, :num_o2o])
        if num_o2m > 0:
            _update_out(out, "pred_logits_o2m", outputs_class[:, :, num_o2o:])
            _update_out(out, "pred_boxes_o2m", outputs_coord[:, :, num_o2o:])
            _update_out(out, "pred_boxes_xyxy_o2m", outputs_boxes_xyxy[:, :, num_o2o:])

    def _run_segmentation_heads(
        self,
        out,
        backbone_out,
        img_ids,
        vis_feat_sizes,
        encoder_hidden_states,
        prompt,
        prompt_mask,
        hs,
        apply_dac=None,
    ):
        # Apply segmentation head (w/ bfloat16 autocast just like the rest of the model)
        apply_dac = apply_dac if apply_dac is not None else self.transformer.decoder.dac
        if self.segmentation_head is not None:
            num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)
            num_o2m = hs.size(2) - num_o2o
            obj_queries = hs if self.o2m_mask_predict else hs[:, :, :num_o2o]
            seg_head_outputs = activation_ckpt_wrapper(self.segmentation_head)(
                backbone_feats=backbone_out["backbone_fpn"],
                obj_queries=obj_queries,
                image_ids=img_ids,
                encoder_hidden_states=encoder_hidden_states,
                act_ckpt_enable=False,
                prompt=prompt,
                prompt_mask=prompt_mask,
            )
            aux_masks = False  # self.aux_loss and self.segmentation_head.aux_masks
            for k, v in seg_head_outputs.items():
                if k in self.segmentation_head.instance_keys:
                    _update_out(out, k, v[:, :num_o2o], auxiliary=aux_masks)
                    if (
                        self.o2m_mask_predict and num_o2m > 0
                    ):  # handle o2m mask prediction
                        _update_out(
                            out, f"{k}_o2m", v[:, num_o2o:], auxiliary=aux_masks
                        )
                else:
                    out[k] = v

    def _get_best_mask(self, out):
        prev_mask_idx = out["pred_logits"].argmax(dim=1).squeeze(1)
        batch_idx = torch.arange(
            out["pred_logits"].shape[0], device=prev_mask_idx.device
        )
        prev_mask_pred = out["pred_masks"][batch_idx, prev_mask_idx][:, None]
        # Downsample mask to match image resolution.
        prev_mask_pred = self.geometry_encoder.mask_encoder.mask_downsampler(
            prev_mask_pred
        )
        prev_mask_pred = prev_mask_pred.flatten(-2).permute(2, 0, 1)

        return prev_mask_pred

    def forward_video_grounding(
        self,
        backbone_out,
        find_input,
        find_target,
        frame_idx,
        previous_stages_out,
        geometric_prompt: Prompt = None,
        run_encoder: bool = True,
        prev_encoder_out: dict = None,
        visual_prompt=None,
        visual_prompt_mask=None,
        is_instance_prompt=False,
        act_ckpt_enable: bool = False,
        track_in_reverse: bool = False,  # track in reverse time order (for demo usage)
        multimask_output: bool = False,
        prev_mask_pred: torch.Tensor = None,
    ):
        """Only activation checkpointing the inner part of video grounding forward."""
        num_prompts = find_input.img_ids.size(0)
        prev_frame_idx = frame_idx + 1 if track_in_reverse else frame_idx - 1

        prev_tracking_queries = self._init_tracking_queries(
            B=num_prompts,
            is_instance_prompt=is_instance_prompt,
            multimask_output=multimask_output,
        )

        prompt, prompt_mask, backbone_out = self._encode_prompt(
            backbone_out,
            find_input,
            geometric_prompt,
            visual_prompt_embed=visual_prompt,
            visual_prompt_mask=visual_prompt_mask,
            prev_mask_pred=prev_mask_pred,
        )
        # Run the encoder
        if run_encoder:
            backbone_out, encoder_out, _ = self._run_encoder(
                backbone_out,
                find_input,
                prompt,
                prompt_mask,
                prev_mask_pred=prev_mask_pred,
            )
        else:
            assert (
                prev_encoder_out is not None
            ), "Something went wrong. If `run_encoder` is False, encoder outputs from previous step should be passed."
            backbone_out, encoder_out = (
                prev_encoder_out["backbone_out"],
                prev_encoder_out["encoder_out"],
            )

        img_feat_inds = encoder_out["img_feat_inds"]
        # (Note that here we directly index the encoder output visual features into per-query
        # feature maps, so the batch size in decoder will always be the number of text prompts
        # rather than the number of images even under late fusion. This is because for video
        # tracking, the tracking queries will have batch size equal to the number of text prompts
        # anyway. So there is no way to reduce the decoder batch size to be the number of images
        # under late fusion, which is unlike the case of image grounding.)
        encoder_hidden_states = encoder_out["encoder_hidden_states"][:, img_feat_inds]
        pos_embed = encoder_out["pos_embed"][:, img_feat_inds]
        assert encoder_hidden_states.size(1) == num_prompts
        assert pos_embed.size(1) == num_prompts
        src_mask = None
        if encoder_out["padding_mask"] is not None:
            src_mask = encoder_out["padding_mask"][img_feat_inds]  # mask is batch-first
            assert src_mask.size(0) == num_prompts

        out = {
            "encoder_hidden_states": encoder_hidden_states,
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }
        # Run the decoder
        out, hs = self._run_decoder_for_tracking(
            memory=encoder_hidden_states,
            pos_embed=pos_embed,
            src_mask=src_mask,
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
            tracking_queries=prev_tracking_queries,
            is_instance_prompt=is_instance_prompt,
            is_multimask_output=multimask_output,
            apply_dac=False,
        )

        # Run segmentation heads
        self._run_segmentation_heads(
            out=out,
            backbone_out=backbone_out,
            img_ids=find_input.img_ids,
            vis_feat_sizes=encoder_out["vis_feat_sizes"],
            encoder_hidden_states=encoder_hidden_states,
            prompt=prompt,
            prompt_mask=prompt_mask,
            hs=hs,
            apply_dac=False,
        )

        out = self._postprocess_out(out, multimask_output=multimask_output)
        return out, backbone_out

    def _postprocess_out(self, out: Dict, multimask_output: bool = False):
        # TODO: Drop some keys to save memory
        # For multimask output, during eval we return the single best mask with the
        # dict keys expected by the evaluators, but also return the multimasks output with new keys.
        num_mask_preds = out["pred_masks"].size(1)
        if not self.training and multimask_output and num_mask_preds > 1:
            out["multi_pred_logits"] = out["pred_logits"]
            out["multi_pred_masks"] = out["pred_masks"]
            out["multi_pred_boxes"] = out["pred_boxes"]
            out["multi_pred_boxes_xyxy"] = out["pred_boxes_xyxy"]

            best_mask_idx = out["pred_logits"].argmax(1).squeeze(1)
            batch_idx = torch.arange(len(best_mask_idx), device=best_mask_idx.device)

            out["pred_logits"] = out["pred_logits"][batch_idx, best_mask_idx].unsqueeze(
                1
            )
            out["pred_masks"] = out["pred_masks"][batch_idx, best_mask_idx].unsqueeze(1)
            out["pred_boxes"] = out["pred_boxes"][batch_idx, best_mask_idx].unsqueeze(1)
            out["pred_boxes_xyxy"] = out["pred_boxes_xyxy"][
                batch_idx, best_mask_idx
            ].unsqueeze(1)

        return out

    def _run_decoder_for_tracking(
        self,
        pos_embed,
        memory,
        src_mask,
        out,
        prompt,
        prompt_mask,
        encoder_out,
        tracking_queries,
        apply_dac=None,
        **kwargs,
    ):
        # In Video OWL-ViT style tracking, we directly feed previous frame's decoder
        # output embeddings from as current frame's decoder inputs.
        tgt = tracking_queries["embed"]
        reference_boxes = tracking_queries["reference_boxes"]

        hs, reference_boxes = self.transformer.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=src_mask,
            pos=pos_embed,
            reference_boxes=reference_boxes,
            level_start_index=encoder_out["level_start_index"],
            spatial_shapes=encoder_out["spatial_shapes"],
            valid_ratios=encoder_out["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=apply_dac,
        )
        hs = hs.transpose(1, 2)  # seq-first to batch-first
        reference_boxes = reference_boxes.transpose(1, 2)  # seq-first to batch-first
        self._update_scores_and_boxes(
            out, hs, reference_boxes, prompt, prompt_mask, apply_dac
        )
        # in Video OWL-ViT style tracking, all output queries are valid
        scores = out["pred_logits"].squeeze(-1)
        out["pred_is_valid"] = torch.ones_like(scores, dtype=torch.bool)  # (B, Q) shape
        # the previously tracked object ids for all (det + track) output queries
        out["pred_old_obj_ids"] = tracking_queries["object_ids"]
        return out, hs

    def _init_tracking_queries(self, B, is_instance_prompt, multimask_output=False):
        """Initialize the tracking queries for the first frame of a video."""
        # Following Video OWL-ViT, on the first frame, the tracking queries are initialized
        # using the learned detection queries.
        if is_instance_prompt and self.use_instance_query:
            query_embed = self.transformer.decoder.instance_query_embed.weight
            query_embed = query_embed[1:] if multimask_output else query_embed[:1]
            reference_boxes = self.transformer.decoder.instance_reference_points.weight
            reference_boxes = (
                reference_boxes[1:] if multimask_output else reference_boxes[:1]
            )
        else:
            query_embed = self.transformer.decoder.query_embed.weight
            reference_boxes = self.transformer.decoder.reference_points.weight

        reference_boxes = reference_boxes.unsqueeze(1).expand(-1, B, -1).sigmoid()
        device = query_embed.device
        init_embed = query_embed.unsqueeze(1).expand(-1, B, -1)  # (Q, B, D), seq-first
        # Initial object ids are all -1, meaning that they are not tracking any objects yet
        Q = query_embed.size(0)
        init_obj_ids = -torch.ones(B, Q, dtype=torch.long, device=device)
        # Initialize the keep-alive countdown for each query. If the tracked object of a query
        # goes out of frame or gets occluded, we maintain its tracking object id for this countdown
        # number of frames before resetting its object id to -1.
        keep_alive_countdown = -torch.ones_like(init_obj_ids)  # (B, Q) shape
        tracking_queries = {
            "embed": init_embed,  # (Q, B, D) shape, seq-first
            "reference_boxes": reference_boxes,  # (Q, B, D) shape, seq-first
            "object_ids": init_obj_ids,  # (B, Q) shape
            "keep_alive_countdown": keep_alive_countdown,  # (B, Q) shape
            # the maximum object id assigned so far (to assign new ids during inference)
            "max_object_id": torch.zeros(B, 1, dtype=torch.long, device=device),
        }
        return tracking_queries
