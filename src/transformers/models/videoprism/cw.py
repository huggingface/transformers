import torch
from safetensors.torch import save_file, load_file
from collections import OrderedDict
from transformers import VideoPrismConfig
from transformers import VideoPrismModel
from huggingface_hub import hf_hub_download, HfApi
import numpy as np
import mediapy

#? download and load the orginal weights
def download_weights():
    # Download the weights file
    file = hf_hub_download(
        repo_id="google/videoprism-base-f16r288", filename="flax_base_f16r288_repeated.npz"
    )
    state_dict = np.load(file)
    return state_dict

checkpoint_dict = {}

def transform_state_encoder_block(state):
    #? spatial encoder blocks
    new_state = OrderedDict()
    spatial_prefix = 'params/spatial_encoder/transformers_stack/x_layers'
    temporal_prefix = 'params/temporal_encoder/transformers_stack/x_layers'
    spatial = 'spatial_encoder.layer'
    temporal = 'temporal_encoder.layer'

    for mode in ['spatial', 'temporal']:
        prefix = spatial_prefix if mode == 'spatial' else temporal_prefix
        layer = spatial if mode == 'spatial' else temporal
        num_layers = 12 if mode == 'spatial' else 4

        for i in range(num_layers):
            #? attention LN
            new_state[f'{layer}.{i}.layernorm_before.weight'] = state[f'{prefix}/layer_norm/scale'][i]  #? [768]
            new_state[f'{layer}.{i}.layernorm_before.bias'] = state[f'{prefix}/layer_norm/bias'][i]  #? [768]
            #? attention
            new_state[f'{layer}.{i}.attention.attention.query.weight'] = state[f'{prefix}/self_attention/query/w'][i].reshape(768, -1).T #? [768, 12, 64] -> [768, 768]
            new_state[f'{layer}.{i}.attention.attention.query.bias'] = state[f'{prefix}/self_attention/query/b'][i].reshape(-1)
            new_state[f'{layer}.{i}.attention.attention.key.weight'] = state[f'{prefix}/self_attention/key/w'][i].reshape(768, -1).T #? [768, 12, 64] -> [768, 768]
            new_state[f'{layer}.{i}.attention.attention.key.bias'] = state[f'{prefix}/self_attention/key/b'][i].reshape(-1)        
            new_state[f'{layer}.{i}.attention.attention.value.weight'] = state[f'{prefix}/self_attention/value/w'][i].reshape(768, -1).T #? [768, 12, 64] -> [768, 768]
            new_state[f'{layer}.{i}.attention.attention.value.bias'] = state[f'{prefix}/self_attention/value/b'][i].reshape(-1)       
            new_state[f'{layer}.{i}.attention.output.dense.weight'] = state[f'{prefix}/self_attention/post/w'][i].reshape(768, -1) #? [768, 12, 64] -> [768, 768]
            new_state[f'{layer}.{i}.attention.output.dense.bias'] = state[f'{prefix}/self_attention/post/b'][i].reshape(-1)
            #? MLP LN
            new_state[f'{layer}.{i}.layernorm_after.weight'] = state[f'{prefix}/ff_layer/layer_norm/scale'][i] #? [768]
            new_state[f'{layer}.{i}.layernorm_after.bias'] = state[f'{prefix}/ff_layer/layer_norm/bias'][i] #? [768]
            #? MLP
            new_state[f'{layer}.{i}.intermediate.dense.weight'] = state[f'{prefix}/ff_layer/ffn_layer1/linear/kernel'][i].T #? [768, 3072] -> [3072, 768]
            new_state[f'{layer}.{i}.intermediate.dense.bias'] = state[f'{prefix}/ff_layer/ffn_layer1/linear/bias'][i]
            new_state[f'{layer}.{i}.output.dense.weight'] = state[f'{prefix}/ff_layer/ffn_layer2/linear/kernel'][i].T #? [768, 3072] -> [3072, 768]
            new_state[f'{layer}.{i}.output.dense.bias'] = state[f'{prefix}/ff_layer/ffn_layer2/linear/bias'][i] 
    return new_state

def  transform_state(state):
    
    new_state = OrderedDict()

    #? patch embeds
    new_state['spatial_embeddings.patch_embeddings.projection.weight'] = state['params/patch_projection/linear/kernel'].T.reshape(768, 1, 18, 18, 3).transpose(0, 4, 1, 2, 3)  #? [972, 768] -> [768, 3, 1, 18, 18]
    new_state['spatial_embeddings.patch_embeddings.projection.bias'] = state['params/patch_projection/linear/bias'] #? [768]w
    
    #? Spatial/temporal pos embeds
    new_state['spatial_embeddings.spatial_pos_emb'] = np.expand_dims(state['params/spatial_pos_emb/emb_var'],axis=0) #? [256, 768] -> [1, 256, 768]
    new_state['temporal_embeddings.temporal_pos_emb'] = np.expand_dims(state['params/temporal_pos_emb/emb_var'],axis=0) #? [256, 768] -> [1, 256, 768]

    #? 'pre' layernorm
    new_state['layernorm1.weight'] = state['params/spatial_ln/scale'] #? all 768 
    new_state['layernorm1.bias'] = state['params/spatial_ln/bias']
    new_state['layernorm2.weight'] = state['params/temporal_ln/scale']
    new_state['layernorm2.bias'] = state['params/temporal_ln/bias']

    new_state.update(transform_state_encoder_block(state))

    checkpoint = {k: torch.tensor(v).contiguous() for k, v in new_state.items()}

    save_file(checkpoint, "videoprism_base_f16r288.safetensors", metadata={"format": "safetensors"})
    print("file saved")
    return 

def read_and_preprocess_video(
    filename: str, target_num_frames: int, target_frame_size: tuple[int, int]
    ):
    """Reads and preprocesses a video."""

    frames = mediapy.read_video(filename)

    # Sample to target number of frames.
    frame_indices = np.linspace(
        0, len(frames), num=target_num_frames, endpoint=False, dtype=np.int32
    )
    frames = np.array([frames[i] for i in frame_indices])

    # Resize to target size.
    original_height, original_width = frames.shape[-3:-1]
    target_height, target_width = target_frame_size
    assert (
        original_height * target_width == original_width * target_height
    ), 'Currently does not support aspect ratio mismatch.'
    frames = mediapy.resize_video(frames, shape=target_frame_size)

    # Normalize pixel values to [0.0, 1.0].
    frames = mediapy.to_float01(frames)

    return frames

# 

if __name__ == "__main__":
    # Load the weights
    # state_dict = download_weights()
    # for k, v in state_dict.items():
    #     shape = v.shape
    #     new_shape = ()
    #     for i in range(len(shape)):
    #         new_shape += (shape[i]-1,)
    #     print(f"Key: {k}, Value shape: {shape}, values: {v[new_shape]} ")
    
    # #first = state_dict["params/patch_projection/linear/bias"]
    # checkpoint = transform_state(state_dict)
    # api = HfApi()
    # api.upload_file(
    # path_or_fileobj="videoprism_base_f16r288.safetensors",
    # path_in_repo="videoprism_base_f16r288.safetensors",
    # repo_id="MHRDYN7/videoprism-base",
    # repo_type="model",
    # )
    # print("uploaded")

    model = VideoPrismModel(VideoPrismConfig())
    state_dict = load_file("videoprism_base_f16r288.safetensors")
    # for k, v in state_dict.items():
    #     shape = v.shape
    #     new_shape = ()
    #     for i in range(len(shape)):
    #         new_shape += (shape[i]-1,)
    #     print(f"Key: {k}, Value shape: {shape}, values: {v[new_shape]} ")

    model.load_state_dict(state_dict)
    print("all good")
    VIDEO_FILE_PATH = (
        './src/transformers/models/videoprism/water_bottle_drumming.mp4'
    )
    NUM_FRAMES = 16
    FRAME_SIZE = 288

    frames = read_and_preprocess_video(
        VIDEO_FILE_PATH,
        target_num_frames=NUM_FRAMES,
        target_frame_size=[FRAME_SIZE, FRAME_SIZE],
    )

    inputs = torch.tensor(frames).unsqueeze(0).permute(0, 1, 4, 2, 3)   #? (1, 16, 3, 288, 288)

    #? (1, 16, 3, 288, 288) is the needed 
    # print(f'Input shape: {inputs.shape} and some values: {inputs[0, 0, :, 0, 0]}')
    with torch.no_grad():
        outputs = model(inputs, output_hidden_states=True, output_attentions=True)
    #print(outputs.last_hidden_state.shape)  # Should print the shape of the output tensor
    print(f'Encoded embedding shape: {outputs.last_hidden_state.shape}, and some values: {outputs.last_hidden_state[0, :3, :3]}')
    print(f'{len(outputs.temporal_hidden_states)=}, {outputs.temporal_hidden_states[0].shape=}, {outputs.temporal_hidden_states[0][0, :3, :3]=}')
    print(f'{len(outputs.spatial_hidden_states)=}, {outputs.spatial_hidden_states[0].shape=}, {outputs.spatial_hidden_states[0][0, :3, :3]=}')
    print(f'{len(outputs.temporal_attentions)=}, {outputs.temporal_attentions[0].shape=}, {outputs.temporal_attentions[0][0, :3, :3]=}')
    print(f'{len(outputs.spatial_attentions)=}, {outputs.spatial_attentions[0].shape=}, {outputs.spatial_attentions[0][0, :3, :3]=}')    
    print("Model loaded and ran successfully")
    # '''
    # The next steps are
    # - Run the original model and get the input and ouput tensor shape plus sample values
    # - replicate the input processor
    # - check if input is same
    # - check if the ouput is same, if not fix the model
    # - once everything is ok, congratulate yourself and upload the model to huggingface
    # '''