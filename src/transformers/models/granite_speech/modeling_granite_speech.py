from typing import List, Optional

import torch
import torch.utils.checkpoint

from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.granite import GraniteForCausalLM
from .configuration_granite_speech import GraniteSpeechConfig
from .projector import EncoderProjectorQFormer
from .encoder import CTCModel

from peft import get_peft_model, LoraConfig, TaskType
import time


class GraniteSpeechForConditionalGeneration(PreTrainedModel, GenerationMixin):
    def __init__(self, config: GraniteSpeechConfig):
        super().__init__(config)

        self.llm = GraniteForCausalLM.from_pretrained(config.llm_name)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_modules,
        )
        self.llm = get_peft_model(self.llm, peft_config)

        self.encoder = CTCModel(config.encoder_config)

        self.projector = EncoderProjectorQFormer(config.projector_config)

        encoder_state_dict = torch.load(
            "data/encoder.pt", map_location="cpu", weights_only=True
        )
        print(self.encoder.load_state_dict(encoder_state_dict, strict=False))

        lora_state_dict = torch.load(
            "data/lora_adapter.pt", map_location="cpu", weights_only=True
        )
        self.llm.load_state_dict(lora_state_dict, strict=False)

        projector_state_dict = torch.load(
            "data/projector.pt", map_location="cpu", weights_only=True
        )
        self.projector.load_state_dict(projector_state_dict, strict=True)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_features: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ): 
        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_for_generation(
                input_ids=input_ids,
                input_features=input_features,
                attention_mask=attention_mask,
            )
        llm_outputs = self.llm(inputs_embeds=inputs_embeds, 
                               attention_mask=attention_mask,
                               past_key_values=past_key_values,
                               position_ids=position_ids,
                               labels=labels, 
                               use_cache=use_cache,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states, 
                               return_dict=return_dict,

                               )
        return llm_outputs

    def generate(
        self,
        input_ids,
        inputs_embeds=None,
        input_features=None,
        attention_mask=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_for_generation(
                input_ids=input_ids,
                input_features=input_features,
                attention_mask=attention_mask,
            )
        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )
        return model_outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        input_features=None,
        **kwargs,
    ):
        a = time.time()
        encoder_embeds = self.encoder(input_features)
        print("Encoder", time.time() - a, "secs")

        a = time.time()
        projected_embeds = self.projector(encoder_embeds, None)
        print("Projector", time.time() - a, "secs")

        a = time.time()
        # concatenate embeddings and invoke LLM generate
        # tokenizer.vocab[self_processor.audio_token]
        combined_embeds = self.get_merged_audio_embeddings(
            input_ids=input_ids,
            audio_features=projected_embeds,
        )
        return combined_embeds

    def get_merged_audio_embeddings(self, input_ids, audio_features):
        """
        Adds the audio token to the model's LLM vocabulary so that we can pass it
        through the tokenizer; it's assumed that the embeddings corresponding to the
        <|audio|> token will be clobbered with speech features.

        TODO - This needs to be adapted to handle batches of variable length sequences
        and potentially labels.
        """
        is_audio_index = input_ids == self.config.audio_token_index
        llm_input_ids = torch.where(is_audio_index, 0, input_ids)
        inputs_embeds = self.llm.get_input_embeddings()(
            llm_input_ids
        )  # [bsz, # features, hidden size]

        # Mask the audio features into the text embeddings
        special_audio_mask = is_audio_index.unsqueeze(-1)
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(
            special_audio_mask,
            audio_features,
        )
        return inputs_embeds

__all__ = ["GraniteSpeechForConditionalGeneration"]
