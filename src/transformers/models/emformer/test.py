import torch

from transformers import EmformerFeatureExtractor, EmformerForRNNT, EmformerTokenizer


feature_extractor = EmformerFeatureExtractor.from_pretrained("anton-l/emformer-base-librispeech")
tokenizer = EmformerTokenizer.from_pretrained("anton-l/emformer-base-librispeech")
model = EmformerForRNNT.from_pretrained("anton-l/emformer-base-librispeech")
model.eval()

waveform = torch.load("/home/anton/repos/audio/examples/asr/emformer_rnnt/librispeech_waveform_0.pt")

features = feature_extractor(waveform, return_tensors="pt")
with torch.no_grad():
    outputs = model(**features)
    token_ids = torch.stack(outputs.logits, dim=-2).squeeze(0).argmax(-1)
    print(tokenizer.decode(token_ids))
