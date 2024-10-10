'''Unit tests for a sentiment analysis model '''
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import unittest


#TestSentimentModel class inherits from unittest.TestCase, providing various testing functionalities.
class TestSentimentModel(unittest.TestCase):

    @classmethod


    #Initializing the tokenizer and model, loading the pre-trained DistilBERT model for sequence classification with three output labels (for positive, neutral, and negative sentiments).
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        cls.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)


    #to check if both the tokenizer and the model are successfully loaded.
    def test_model_initialization(self):
        self.assertIsNotNone(self.tokenizer)
        self.assertIsNotNone(self.model)

    #Tests the model's ability to make a sentiment prediction on a single input
    def test_inference_single_example(self):
        test_text = ["Sepal Length: 5.1, Sepal Width: 3.5, Petal Length: 1.4, Petal Width: 0.2"]
        test_encodings = self.tokenizer(test_text, truncation=True, padding=True, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**test_encodings)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        self.assertIn(prediction, [0, 1, 2])

    #Tests the model's ability to process a batch of inputs and return predictions.
    def test_batch_inference(self):
        test_texts = [
            "Sepal Length: 5.1, Sepal Width: 3.5, Petal Length: 1.4, Petal Width: 0.2",
            "Sepal Length: 6.0, Sepal Width: 2.2, Petal Length: 5.0, Petal Width: 1.5"
        ]
        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**test_encodings)
            predictions = torch.argmax(outputs.logits, dim=1).numpy()

        for pred in predictions:
            self.assertIn(pred, [0, 1, 2])

    #To checks if the model outputs the correct shape for a single input.
    def test_model_output_shape(self):
        test_text = ["Sepal Length: 5.1, Sepal Width: 3.5, Petal Length: 1.4, Petal Width: 0.2"]
        test_encodings = self.tokenizer(test_text, truncation=True, padding=True, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**test_encodings)

        self.assertEqual(outputs.logits.shape, (1, 3))  # 1 example, 3 possible labels

    #To Validate the tokenization process.
    def test_tokenization(self):
        test_text = ["Sepal Length: 5.1, Sepal Width: 3.5, Petal Length: 1.4, Petal Width: 0.2"]
        test_encodings = self.tokenizer(test_text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')

        self.assertEqual(test_encodings['input_ids'].shape[1], 128)

    #To test the model's behavior with an empty input.
    def test_invalid_input(self):
        test_text = [""]
        test_encodings = self.tokenizer(test_text, truncation=True, padding=True, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(**test_encodings)
        
        #To check if the output shape is correct (1 example, num_labels)
        self.assertEqual(outputs.logits.shape, (1, 3))  # Assuming 3 labels
        #To check if that logits are not all the same (as they shouldn't be for empty input)
        self.assertFalse(torch.all(outputs.logits == outputs.logits[0, 0]))  # Not all logits should be the same


    #To ensure that the model produces consistent predictions for the same input on multiple runs.
    def test_prediction_consistency(self):
        test_text = ["Sepal Length: 5.1, Sepal Width: 3.5, Petal Length: 1.4, Petal Width: 0.2"]
        test_encodings = self.tokenizer(test_text, truncation=True, padding=True, return_tensors='pt')
        
        with torch.no_grad():
            outputs_1 = self.model(**test_encodings)
            outputs_2 = self.model(**test_encodings)

        self.assertTrue(torch.equal(outputs_1.logits, outputs_2.logits))


#To run the test
if __name__ == '__main__':
    unittest.main()
