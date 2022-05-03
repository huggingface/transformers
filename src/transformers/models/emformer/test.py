from transformers import EmformerFeatureExtractor


feature_extractor = EmformerFeatureExtractor()
feature_extractor.save_pretrained("./emformer-base-librispeech")
