from transformers.modeling_bert import *
from gQA.model_code.gnn_bridge import *
from gQA.model_code.state import State
import sys

class BertForQuestionAnswering_GNN(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForQuestionAnswering_GNN, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.bridge = GNN_Bridge(config.hidden_size, 48, config)
        d_hs = config.hidden_size + 0
        #Linear map for word embeddings
        self.L = nn.Linear(config.hidden_size, d_hs)
        #Linear map for enriched sentence embeddings
        self.W = nn.Linear(config.hidden_size, d_hs)
        #Linear map for the enriched question embedding
        self.H = nn.Linear(config.hidden_size, d_hs)
        #Linear map for the enriched document embedding
        self.G = nn.Linear(config.hidden_size, d_hs)
        #Combine the above embeddings
        self.qa_outputs = nn.Linear(d_hs, config.num_labels)
        
        self.config = config
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, adjm=None, 
    sent_attn_mask=None, token_type_ids=None, position_ids=None, 
    head_mask=None, start_positions=None, end_positions=None,
    addrx=None, addry=None, device=None):

        _, batch_size, docu_len, sent_len = input_ids.size()

        input_ids = input_ids.view(batch_size * docu_len, sent_len)
        attention_mask = attention_mask.view(batch_size * docu_len, sent_len)
        token_type_ids = token_type_ids.view(batch_size * docu_len, sent_len)
        if position_ids is not None:
            position_ids = position_ids.view(batch_size * docu_len, sent_len)

        adjm = adjm.view(batch_size, docu_len, docu_len)
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=None,
                            head_mask=head_mask)
        #h_word = outputs[0].view(batch_size, docu_len, sent_len, -1)
        h_sent = outputs[1].view(batch_size, docu_len, -1)


        questions = h_sent[0:, 0, 0:]

        bridge_input = State(
            outputs=h_sent,
            mask=sent_attn_mask,
            other=[adjm, device, [batch_size, docu_len, sent_len]])

        doc_embd, h_gcn = self.bridge(bridge_input, questions)

        #Appropriate resizing
        sequence_output = outputs[0]
        questions = questions.unsqueeze(1).expand(batch_size, docu_len, self.config.hidden_size).contiguous()
        questions = questions.view(batch_size * docu_len, 1, -1).expand(sequence_output.size()).contiguous()
        doc_embd = doc_embd.view(batch_size * docu_len, 1, -1).expand(sequence_output.size()).contiguous()
        h_gcn = h_gcn.view(batch_size * docu_len, 1, -1).expand(sequence_output.size()).contiguous()

        #Compute logits
        logits = self.qa_outputs(self.L(sequence_output) + self.G(doc_embd))
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        #Resize to acocunt for sentence length
        start_logits = start_logits.view(batch_size, docu_len * sent_len)
        end_logits = end_logits.view(batch_size, docu_len * sent_len)

        #start_logits = start_logits.view(batch_size, docu_len,)
        outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            
            addrx = addrx.squeeze(0)
            addry = addry.squeeze(0)
            start_positions = start_positions.squeeze().to(device)
            end_positions = end_positions.squeeze().to(device)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)