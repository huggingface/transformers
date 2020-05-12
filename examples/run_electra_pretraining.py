from transformers import ElectraForPreTraining, ElectraForMaskedLM, ElectraTokenizerFast, PreTrainedModel
import torch
import torch.nn as nn


class CombinedModel(nn.Module):
    def __init__(self,  discriminator: PreTrainedModel, generator: PreTrainedModel):
        super().__init__()

        self.discriminator = discriminator
        self.generator = generator

    @staticmethod
    def mask_inputs(
        input_ids: torch.Tensor,
        mask_token_id,
        mask_probability,
        tokens_to_ignore,
        max_predictions_per_seq,
        proposal_distribution=1.0
    ):
        inputs_which_can_be_masked = torch.ones_like(input_ids)
        for token in tokens_to_ignore:
            inputs_which_can_be_masked -= torch.eq(input_ids, token).long()

        total_number_of_tokens = input_ids.shape[-1]

        # Identify the number of tokens to be masked, which should be: 1 < num < max_predictions per seq.
        # It is set to be: n_tokens * mask_probability, but is truncated if it goes beyond bounds.
        number_of_tokens_to_be_masked = torch.max(
            torch.tensor(1),
            torch.min(
                torch.tensor(max_predictions_per_seq),
                torch.tensor(total_number_of_tokens * mask_probability, dtype=torch.long)
            )
        )

        # The probability of each token being masked
        sample_prob = proposal_distribution * inputs_which_can_be_masked
        sample_prob /= torch.sum(sample_prob)
        # Should be passed through a log function here

        # Weight of each position: 1 the position will be masked, 0 the position won't be masked
        masked_lm_weights = torch.tensor([0] * max_predictions_per_seq, dtype=torch.bool)
        masked_lm_weights[:number_of_tokens_to_be_masked] = True

        # Sample from the probabilities
        masked_lm_positions = sample_prob.multinomial(max_predictions_per_seq)

        # Apply the weights to the positions
        masked_lm_positions *= masked_lm_weights.long()

        # Gather the IDs from the positions
        masked_lm_ids = input_ids.gather(-1, masked_lm_positions)

        # Apply weights to the IDs
        masked_lm_ids *= masked_lm_weights.long()

        replace_with_mask_positions = masked_lm_positions * (torch.rand(masked_lm_positions.shape) < 0.85)

        # Replace the input IDs with masks on given positions
        masked_input_ids = input_ids.scatter(-1, replace_with_mask_positions, mask_token_id)

        # Updates to index 0 should be ignored
        masked_input_ids[..., 0] = input_ids[..., 0]

        return masked_input_ids, masked_lm_positions

    @staticmethod
    def gather_positions(
        sequence,
        positions
    ):
        batch_size, sequence_length, dimension = sequence.shape
        position_shift = (sequence_length * torch.arange(batch_size)).unsqueeze(-1)
        flat_positions = torch.reshape(positions + position_shift, [-1]).long()
        flat_sequence = torch.reshape(sequence, [batch_size * sequence_length, dimension])
        gathered = flat_sequence.index_select(0, flat_positions)
        return torch.reshape(gathered, [batch_size, -1, dimension])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        masked_input_ids, masked_lm_positions = self.mask_inputs(
            input_ids,
            tokenizer.mask_token_id,
            0.2,
            [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id],
            30
        )

        generator_loss, generator_output = self.generator(
            masked_input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            position_ids,
            masked_lm_labels=input_ids
        )[:2]

        fake_logits = self.gather_positions(generator_output, masked_lm_positions)
        fake_argmaxes = fake_logits.argmax(-1)
        fake_tokens = masked_input_ids.scatter(-1, masked_lm_positions, fake_argmaxes)
        fake_tokens[:, 0] = input_ids[:, 0]

        # discriminator_output
        discriminator_loss, discriminator_output = self.discriminator(
            fake_tokens,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            position_ids,
            labels=input_ids
        )[:2]

        discriminator_predictions = torch.round((torch.sign(discriminator_output) + 1) / 2).int().tolist()

        total_loss = discriminator_loss + generator_loss

        return (
            total_loss,
            (discriminator_predictions, generator_output),
            (fake_tokens, masked_input_ids)
        )


tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-generator")
_generator = ElectraForMaskedLM.from_pretrained("google/electra-small-generator")
_discriminator = ElectraForPreTraining.from_pretrained("google/electra-small-discriminator")
model = CombinedModel(_discriminator, _generator)

text = "Still leaning against the incubators he gave them, while the pencils scurried illegibly across the pages, a brief description of the modern fertilizing process;"
tokens = tokenizer.tokenize(text)

input_ids = tokenizer.batch_encode_plus([text, text], return_tensors="pt")["input_ids"]

loss, predictions, ids = model(input_ids)

discriminator_predictions = predictions[0]
fake_tokens, masked_input_ids = ids

for batch in range(input_ids.shape[0]):
    print(tokenizer.decode(input_ids.tolist()[batch]))
    print(tokenizer.decode(masked_input_ids.tolist()[batch]))
    print(tokenizer.decode(fake_tokens.tolist()[batch]))
    [print("%15s" % token, end="") for token in tokenizer.tokenize(tokenizer.decode(masked_input_ids.tolist()[batch]))]
    print()
    [print("%15s" % prediction, end="") for prediction in discriminator_predictions[0]]
    print("\n\n")


