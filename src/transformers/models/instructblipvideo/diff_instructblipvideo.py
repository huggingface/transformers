# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss

from transformers.models.instructblip.configuration_instructblip import (
    InstructBlipConfig,
    InstructBlipQFormerConfig,
    InstructBlipVisionConfig,
)
from transformers.models.instructblip.modeling_instructblip import (
    InstructBlipAttention,
    InstructBlipEncoder,
    InstructBlipEncoderLayer,
    InstructBlipForConditionalGeneration,
    InstructBlipForConditionalGenerationModelOutput,
    InstructBlipMLP,
    InstructBlipPreTrainedModel,
    InstructBlipQFormerAttention,
    InstructBlipQFormerEmbeddings,
    InstructBlipQFormerEncoder,
    InstructBlipQFormerIntermediate,
    InstructBlipQFormerLayer,
    InstructBlipQFormerModel,
    InstructBlipQFormerOutput,
    InstructBlipQFormerSelfOutput,
    InstructBlipVisionEmbeddings,
    InstructBlipVisionModel,
)

from ...utils import logging


logger = logging.get_logger(__name__)


class InstructBlipVideoVisionConfig(InstructBlipVisionConfig):
    pass


class InstructBlipVideoQFormerConfig(InstructBlipQFormerConfig):
    pass


class InstructBlipVideoConfig(InstructBlipConfig):
    pass


@dataclass
class InstructBlipVideoForConditionalGenerationModelOutput(InstructBlipForConditionalGenerationModelOutput):
    pass


class InstructBlipVideoVisionEmbeddings(InstructBlipVisionEmbeddings):
    pass


class InstructBlipVideoAttention(InstructBlipAttention):
    pass


class InstructBlipVideoMLP(InstructBlipMLP):
    pass


class InstructBlipVideoEncoderLayer(InstructBlipEncoderLayer):
    pass


class InstructBlipVideoPreTrainedModel(InstructBlipPreTrainedModel):
    pass


class InstructBlipVideoEncoder(InstructBlipEncoder):
    pass


class InstructBlipVideoVisionModel(InstructBlipVisionModel):
    pass


class InstructBlipVideoQFormerSelfOutput(InstructBlipQFormerSelfOutput):
    pass


class InstructBlipVideoQFormerAttention(InstructBlipQFormerAttention):
    pass


class InstructBlipVideoQFormerIntermediate(InstructBlipQFormerIntermediate):
    pass


class InstructBlipVideoQFormerOutput(InstructBlipQFormerOutput):
    pass


class InstructBlipVideoQFormerLayer(InstructBlipQFormerLayer):
    pass


class InstructBlipVideoQFormerEncoder(InstructBlipQFormerEncoder):
    pass


class InstructBlipVideoQFormerEmbeddings(InstructBlipQFormerEmbeddings):
    pass


class InstructBlipVideoQFormerModel(InstructBlipQFormerModel):
    pass


class InstructBlipVideoForConditionalGeneration(InstructBlipForConditionalGeneration):
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Union[Tuple, InstructBlipVideoForConditionalGenerationModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
            1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import InstructBlipVideoProcessor, InstructBlipVideoForConditionalGeneration
        >>> import torch
        >>> from huggingface_hub import hf_hub_download
        >>> from av

        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`List[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])

        >>> model = InstructBlipVideoProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b", device_map="auto")
        >>> processor = InstructBlipVideoForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")

        >>> file_path = hf_hub_download(
                repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
            )
        >>> container = av.open(video_path)
        >>> # sample uniformly 4 frames from the videWhy is this video funny?o
        >>> total_frames = container.streams.video[0].frames
        >>> indices = np.arange(0, total_frames, total_frames / 4).astype(int)
        >>> clip = read_video_pyav(container, indices)

        >>> prompt = "What is happening in the video?"
        >>> inputs = processor(videos=clip, text=prompt, return_tensors="pt").to(device)

        >>> outputs = model.generate(
        ...     **inputs,
        ...     do_sample=False,
        ...     num_beams=5,
        ...     max_length=256,
        ...     repetition_penalty=1.5,
        ...     length_penalty=1.0,
        ... )
        >>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        "A person is eating a bowl of pasta, and they are using a fork to eat it. The person is sitting at a table, and the plate of pasta is on the table in front"
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # we process in a batched way, later unbatch it back (video has frames=4 always)
        batch_size, frames, channel, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(batch_size * frames, channel, height, width)

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)

        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)

        qformer_input_ids = qformer_input_ids.repeat_interleave(frames, dim=0)
        qformer_attention_mask = qformer_attention_mask.repeat_interleave(frames, dim=0)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)

        # unbatch inputs back, each video-frame gets `num_query_tokens` seq length
        language_model_inputs = language_model_inputs.reshape(batch_size, self.config.num_query_tokens * frames, -1)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="mean")

                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return InstructBlipVideoForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
           pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width) or
                (batch_size, num_frames, num_channels, height, width)): Input images or videos to be processed.
            qformer_input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt to be fed to the Q-Former module.
            qformer_attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices.
            interpolate_pos_encoding (`bool`, *optional*, defaults to `False`):
                Whether to interpolate the positional encoding of the image embeddings.

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        # we process in a batched way, later unbatch it back (video has frames=4)
        batch_size, frames, channel, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(batch_size * frames, channel, height, width)

        image_embeds = self.vision_model(
            pixel_values,
            return_dict=True,
            interpolate_pos_encoding=interpolate_pos_encoding,
        ).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)

        qformer_input_ids = qformer_input_ids.repeat_interleave(frames, dim=0)
        qformer_attention_mask = qformer_attention_mask.repeat_interleave(frames, dim=0)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]

        language_model_inputs = self.language_projection(query_output)

        # unbatch the embeddings back by moving frames to seq-len
        language_model_inputs = language_model_inputs.reshape(batch_size, self.config.num_query_tokens * frames, -1)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        # add image_embeds length to max_length, so that the final max_length in counted only on token embeds
        # -1 is to account for the prepended BOS after `generate.`
        if not self.language_model.config.is_encoder_decoder:
            generate_kwargs["max_length"] = generate_kwargs.get("max_length", 20) + language_model_inputs.shape[1] - 1
            generate_kwargs["min_length"] = generate_kwargs.get("min_length", 0) + language_model_inputs.shape[1]

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        # this is a temporary workaround to be consistent with other generation models and
        # have BOS as the first token, even though under the hood we are calling LM with embeds
        if not self.language_model.config.is_encoder_decoder:
            # the InstructBLIP authors used inconsistent tokenizer/model files during training,
            # with the tokenizer's bos token being set to </s> which has ID=2,
            # whereas the model's text config has bos token id = 0
            bos_token_id = (
                2
                if self.config.text_config.architectures[0] == "LLaMAForCausalLM"
                else self.config.text_config.bos_token_id
            )
            bos_tokens = torch.LongTensor([[bos_token_id]]).repeat(batch_size, 1).to(image_embeds.device)
            if not isinstance(outputs, torch.Tensor):
                outputs.sequences = torch.cat([bos_tokens, outputs.sequences], dim=-1)
            else:
                outputs = torch.cat([bos_tokens, outputs], dim=-1)

        return outputs
