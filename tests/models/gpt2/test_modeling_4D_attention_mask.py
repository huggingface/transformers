import unittest
import torch
from transformers import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

class TestAttentionMaskIssue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = "gpt2"
        cls.model = GPT2LMHeadModel.from_pretrained(cls.model_name)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)

    def prepare_data(self, packing=False):
        texts = [
            "Hello, how are you?",
            "When is the next holiday?",
            "China is a great country.",
        ]
        encoded = self.tokenizer(texts)
        
        if packing:
            total_length = sum(len(x) for x in encoded["input_ids"])
            input_ids = torch.zeros((1, total_length), dtype=torch.long)
            # Create 4D attention mask with proper shape
            attention_mask = torch.full(
                (1, 1, total_length, total_length), 
                dtype=torch.float32, 
                fill_value=float("-inf")
            )

            offset = 0
            for i, (ids, mask) in enumerate(zip(encoded["input_ids"], encoded["attention_mask"])):
                length = len(ids)
                input_ids[0, offset:offset + length] = torch.tensor(ids)
                # Set valid attention positions to 0
                attention_mask[0, 0, offset:offset + length, :offset + length] = 0.
                offset += length
            
            return input_ids, attention_mask
        else:
            # Regular batched processing
            max_length = max(len(x) for x in encoded["input_ids"])
            input_ids = torch.zeros((len(texts), max_length), dtype=torch.long)
            attention_mask = torch.zeros((len(texts), max_length), dtype=torch.long)
            
            for i, (ids, mask) in enumerate(zip(encoded["input_ids"], encoded["attention_mask"])):
                input_ids[i, :len(ids)] = torch.tensor(ids)
                attention_mask[i, :len(mask)] = torch.tensor(mask)
                
            return input_ids, attention_mask
    
    def test_attention_mask_shapes(self):
        # Test both regular and packed versions
        input_ids_regular, mask_regular = self.prepare_data(packing=False)
        output_regular = self.model(input_ids=input_ids_regular, attention_mask=mask_regular)
        
        input_ids_packed, mask_packed = self.prepare_data(packing=True)
        output_packed = self.model(input_ids=input_ids_packed, attention_mask=mask_packed)
        
        # Verify outputs have expected shapes
        self.assertEqual(
            output_regular.logits.shape[:-1],
            input_ids_regular.shape
        )
        self.assertEqual(
            output_packed.logits.shape[:-1],
            input_ids_packed.shape
        )