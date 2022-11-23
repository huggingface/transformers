import torch.nn as nn
import torch
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from transformers.models.roberta.modeling_roberta import RobertaLMHead,RobertaPreTrainedModel
from transformers.activations import ACT2FN
from typing import Optional
from .configuration_altclip import RobertaSeriesConfig


class RobertaSeriesModelWithTransformation(RobertaPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    config_class = RobertaSeriesConfig

    def __init__(self, config):
        super().__init__(config)
        self.roberta = XLMRobertaModel(config)
        self.transformation = nn.Linear(config.hidden_size,config.project_dim)
        self.pre_LN = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # we turn True only when we do postkd, otherwise it should be turn False 
        self.add_lm_task = config.add_lm_task if hasattr(config,"add_lm_task") else False
        self.config.tie_word_embeddings = self.add_lm_task
        if self.add_lm_task:
            self.lm_head = RobertaLMHead(config)
        self.pooler = lambda x: x[:,0]
        self.post_init()
        
    def get_input_embeddings(self) -> nn.Module:
        return self.roberta.embeddings.word_embeddings
    
    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head.decoder
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        mode: Optional[str] = 'kd',
    ) :
        r"""
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # last module outputs
        sequence_output = outputs[0]

        # project every module
        sequence_output = self.pre_LN(sequence_output)
            
        # pooler
        pooler_output = self.pooler(sequence_output)
        pooler_output = self.transformation(pooler_output)
        projection_state = self.transformation(outputs.last_hidden_state)
        
        mlm_loss = None
        if labels is not None and mode == 'lm' and self.add_lm_task:
            prediction_scores = self.lm_head(sequence_output)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                mlm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            else:
                raise ValueError

        return {
            'pooler_output':pooler_output,
            'last_hidden_state':outputs.last_hidden_state,
            'hidden_states':outputs.hidden_states,
            'attentions':outputs.attentions,
            'projection_state':projection_state,
            'mlm_loss':mlm_loss
        }

