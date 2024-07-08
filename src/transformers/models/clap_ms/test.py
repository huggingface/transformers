"""
This is an example using CLAP to perform zeroshot
    classification on ESC50 (https://github.com/karolpiczak/ESC-50).
"""

from clap_wrapper import CLAPWrapper
import torch.nn.functional as F
import torch

# Load model (Choose between versions '2022' or '2023')
# The model weight will be downloaded automatically if `model_fp` is not specified
clap_model = CLAPWrapper(version = '2023', use_cuda=False)

# Load dataset
# root_path = "/home/kamil/projects/CLAP/examples/root_path" # Folder with ESC-50-master/
# dataset = ESC50(root=root_path, download=False) #If download=False code assumes base_folder='ESC-50-master' in esc50_dataset.py

# # Generate text prompts: 
# prompt = 'this is the sound of '
# class_labels = [prompt + x for x in dataset.classes[:5]]

# # Get audio input prompts: 
# y_preds, y_labels = [], []
# for i in tqdm(range(2)):
#     x, _, one_hot_target = dataset.__getitem__(i)
#     break 

# Feature extractors: 

# audio_feature_extractor = AudioFeatureExtractor(version = '2023')

# preprocessed_text = clap_model.preprocess_text(class_labels)

# preprocessed_audio = clap_model.preprocess_audio([x], resample=True)
# preprocessed_audio = preprocessed_audio.reshape(
#         preprocessed_audio.shape[0], preprocessed_audio.shape[2])

# preprocessed_audio = audio_feature_extractor.forward(preprocessed_audio)


preprocessed_text = torch.load('/home/kamil/transformers/src/transformers/models/clap_ms/preprocessed_text.pth')
preprocessed_audio = torch.load('/home/kamil/transformers/src/transformers/models/clap_ms/preprocessed_audio.pth')

# forward pass for the text model: 
with torch.no_grad():
    text_embeddings = clap_model.clap.caption_encoder(preprocessed_text)
    
    audio_embeddings = clap_model.clap.audio_encoder(preprocessed_audio)[0]


similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()

print(y_pred)


