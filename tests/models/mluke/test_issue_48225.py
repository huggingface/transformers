import unittest
from transformers import MLukeTokenizer

class MLukeTokenizerIssueTest(unittest.TestCase):
    def test_entity_classification_markers(self):
        tokenizer = MLukeTokenizer.from_pretrained("studio-ousia/mluke-base", task="entity_classification")
        text = "Beyonce lives in Los Angeles."
        entity_spans = [(0, 7)]
        enc = tokenizer(text, entity_spans=entity_spans)
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        
        # Check that <ent> markers are present twice
        self.assertEqual(tokens.count("<ent>"), 2)
        # Check that <s> is only present as BOS
        self.assertEqual(tokens.count("<s>"), 1)

    def test_entity_pair_classification_markers(self):
        tokenizer = MLukeTokenizer.from_pretrained("studio-ousia/mluke-base", task="entity_pair_classification")
        text = "Beyonce lives in Los Angeles."
        entity_spans = [(0, 7), (17, 28)] # "Beyonce", "Los Angeles"
        enc = tokenizer(text, entity_spans=entity_spans)
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        
        # Check that <ent> and <ent2> markers are present twice each
        self.assertEqual(tokens.count("<ent>"), 2)
        self.assertEqual(tokens.count("<ent2>"), 2)
        # Check that <s> is only present as BOS
        self.assertEqual(tokens.count("<s>"), 1)

if __name__ == "__main__":
    unittest.main()
