import torch
from safetensors.torch import save_file, load_file
from collections import OrderedDict
from transformers import VideoPrismConfig
from transformers import VideoPrismModel
from huggingface_hub import hf_hub_download, HfApi
import numpy as np
import mediapy


def get_checkpoint_info(model_type='backbone', model_size = 'base'):
    backbone_base = {
        "model_type": "backbone",
        "model_size": "base",
        "id": "f16r288",
        "repo_id": "google/videoprism-base-f16r288",
        "filename": "flax_base_f16r288_repeated.npz",
        "config": {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_frames": 16,
            "num_spatial_layers": 12,
        },
    }
    backbone_large = {
        "model_type": "backbone",
        "model_size": "large",
        "id": "f8r288",
        "repo_id": "google/videoprism-large-f8r288",
        "filename": "flax_large_f8r288_repeated.npz",
        "config": {
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_frames": 8,
            "num_spatial_layers": 24,
        },
    }
    lvt_base = {
        "model_type": "lvt",
        "model_size": "base",
        "id": "f16r288",
        "repo_id": "google/videoprism-lvt-base-f16r288",
        "filename": "flax_lvt_base_f16r288_repeated.npz",
        "config": {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_attention_heads": 12,
            "num_frames": 16,
            "num_spatial_layers": 12,
        },
    }
    lvt_large = {
        "model_type": "lvt",
        "model_size": "large",
        "id": "f8r288",
        "repo_id": "google/videoprism-lvt-large-f8r288",
        "filename": "flax_lvt_large_f8r288_repeated.npz",
        "config": {
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_frames": 8,
            "num_spatial_layers": 24,
        },
    }
    if model_type == 'backbone':
        return backbone_base if model_size == 'base' else backbone_large
    
    elif model_type == 'lvt':
        return lvt_base if model_size == 'base' else lvt_large

#? download and load the orginal weights
def download_weights(checkpoint_info):
    # Download the weights file
    file = hf_hub_download(
        repo_id=checkpoint_info["repo_id"], filename=checkpoint_info["filename"]
    )
    state_dict = np.load(file)
    return state_dict

checkpoint_dict = {}

def transform_state_encoder_block(state, checkpoint_info):
    #? spatial encoder blocks
    new_state = OrderedDict()
    spatial_prefix = 'params/spatial_encoder/transformers_stack/x_layers'
    temporal_prefix = 'params/temporal_encoder/transformers_stack/x_layers'
    spatial = 'spatial_encoder.layer'
    temporal = 'temporal_encoder.layer'
    num_spatial_layers = checkpoint_info['config']['num_spatial_layers']
    hidden_size = checkpoint_info['config']['hidden_size']

    for mode in ['spatial', 'temporal']:
        prefix = spatial_prefix if mode == 'spatial' else temporal_prefix
        layer = spatial if mode == 'spatial' else temporal
        num_layers = num_spatial_layers if mode == 'spatial' else 4

        for i in range(num_layers):
            #? attention LN
            new_state[f'{layer}.{i}.layernorm_before.weight'] = state[f'{prefix}/layer_norm/scale'][i]  #? [768]
            new_state[f'{layer}.{i}.layernorm_before.bias'] = state[f'{prefix}/layer_norm/bias'][i]  #? [768]
            #? attention
            new_state[f'{layer}.{i}.attention.attention.query.weight'] = state[f'{prefix}/self_attention/query/w'][i].reshape(hidden_size, -1).T #? [768, 12, 64] -> [768, 768]
            new_state[f'{layer}.{i}.attention.attention.query.bias'] = state[f'{prefix}/self_attention/query/b'][i].reshape(-1)
            new_state[f'{layer}.{i}.attention.attention.key.weight'] = state[f'{prefix}/self_attention/key/w'][i].reshape(hidden_size, -1).T #? [768, 12, 64] -> [768, 768]
            new_state[f'{layer}.{i}.attention.attention.key.bias'] = state[f'{prefix}/self_attention/key/b'][i].reshape(-1)        
            new_state[f'{layer}.{i}.attention.attention.value.weight'] = state[f'{prefix}/self_attention/value/w'][i].reshape(hidden_size, -1).T #? [768, 12, 64] -> [768, 768]
            new_state[f'{layer}.{i}.attention.attention.value.bias'] = state[f'{prefix}/self_attention/value/b'][i].reshape(-1)       
            new_state[f'{layer}.{i}.attention.output.dense.weight'] = state[f'{prefix}/self_attention/post/w'][i].reshape(hidden_size, -1) #? [768, 12, 64] -> [768, 768]
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

def  transform_state(state, checkpoint_info):
    hidden_size = checkpoint_info['config']['hidden_size']
    new_state = OrderedDict()

    #? patch embeds
    new_state['spatial_embeddings.patch_embeddings.projection.weight'] = state['params/patch_projection/linear/kernel'].T.reshape(hidden_size, 1, 18, 18, 3).transpose(0, 4, 1, 2, 3)  #? [972, 768] -> [768, 3, 1, 18, 18]
    new_state['spatial_embeddings.patch_embeddings.projection.bias'] = state['params/patch_projection/linear/bias'] #? [768]
    #? Spatial/temporal pos embeds
    new_state['spatial_embeddings.spatial_pos_emb'] = np.expand_dims(state['params/spatial_pos_emb/emb_var'],axis=0) #? [256, 768] -> [1, 256, 768]
    new_state['temporal_embeddings.temporal_pos_emb'] = np.expand_dims(state['params/temporal_pos_emb/emb_var'],axis=0) #? [256, 768] -> [1, 256, 768]
    #? 'pre' layernorm
    new_state['layernorm1.weight'] = state['params/spatial_ln/scale'] #? all 768 
    new_state['layernorm1.bias'] = state['params/spatial_ln/bias']
    new_state['layernorm2.weight'] = state['params/temporal_ln/scale']
    new_state['layernorm2.bias'] = state['params/temporal_ln/bias']

    new_state.update(transform_state_encoder_block(state, checkpoint_info))

    checkpoint = {k: torch.tensor(v).contiguous() for k, v in new_state.items()}

    if checkpoint_info['model_type'] == 'backbone':
        path = f"videoprism_{checkpoint_info['model_size']}_{checkpoint_info['id']}.safetensors"
    elif checkpoint_info['model_type'] == 'lvt':
        path = f"videoprism_lvt_{checkpoint_info['model_size']}_{checkpoint_info['id']}.safetensors"

    save_file(checkpoint, path, metadata={"format": "safetensors"})
    print("file saved")


def prepare_video():
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti_32_frames.npy", repo_type="dataset"
    )
    video = np.load(file)
    return list(video)

    

def read_and_preprocess_video(   # This function from the original code
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

def convert(model_type='backbone', model_size='base', convert=False, upload=False, load_model=True, load_video=True, inference=True):
    # Load the weights
    checkpoint_info = get_checkpoint_info(model_type, model_size)

    if checkpoint_info['model_type'] == 'backbone':
        path = f"videoprism_{checkpoint_info['model_size']}_{checkpoint_info['id']}.safetensors"
    elif checkpoint_info['model_type'] == 'lvt':
        path = f"videoprism_lvt_{checkpoint_info['model_size']}_{checkpoint_info['id']}.safetensors"
    
    
    if convert:
        
        state_dict = download_weights(checkpoint_info)
        for k, v in state_dict.items():
            shape = v.shape
            new_shape = ()
            for i in range(len(shape)):
                new_shape += (shape[i]-1,)
            print(f"Key: {k}, Value shape: {shape}, values: {v[new_shape]} ")
    
        #first = state_dict["params/patch_projection/linear/bias"]
        #transform_state(state_dict, checkpoint_info)

    if upload:
        api = HfApi()
        api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path,
        repo_id="MHRDYN7/videoprism-base",
        repo_type="model",
        )
        print("uploaded")
    
    if load_model:
        config = VideoPrismConfig(**checkpoint_info['config'])
        model = VideoPrismModel(config)
        state_dict = load_file(path)
        # for k, v in state_dict.items():
        #     shape = v.shape
        #     new_shape = ()
        #     for i in range(len(shape)):
        #         new_shape += (shape[i]-1,)
        #     print(f"Key: {k}, Value shape: {shape}, values: {v[new_shape]} ")

        model.load_state_dict(state_dict)
        # print("all good")

    
    if load_video:
        VIDEO_FILE_PATH = (
        './src/transformers/models/videoprism/water_bottle_drumming.mp4'
        )
        NUM_FRAMES = checkpoint_info['config']['num_frames']  #? 16 for base, 8 for large
        FRAME_SIZE = 288
        frames = read_and_preprocess_video(
            VIDEO_FILE_PATH,
            target_num_frames=NUM_FRAMES,
            target_frame_size=[FRAME_SIZE, FRAME_SIZE],
        )

        inputs = torch.tensor(frames).unsqueeze(0).permute(0, 1, 4, 2, 3)   #? (1, 16, 3, 288, 288)
        
        # inputs = prepare_video()
        # frame_indices = np.linspace(
        #     0, len(inputs), num=16, endpoint=False, dtype=np.int32
        # )
        # inputs = np.array([inputs[i] for i in frame_indices])
        # inputs = VideoPrismVideoProcessor()(inputs, return_tensors="pt")
        #? (1, 16, 3, 288, 288) is the needed 

    
    if inference:
        with torch.no_grad():
            outputs = model(inputs, output_hidden_states=True, output_attentions=True)
            backbone_base_expected_tensor = torch.tensor([
                [0.11648951, 0.4568253, 0.19288044],
                [0.28420594, -0.04224018, 0.377879],
                [0.24594213, -0.3914095, -0.30516925]]
                )
            backbone_large_expected_tensor = torch.tensor([
                [0.39503154, 0.07308281, 0.21407786],
                [ 0.4963156, -0.02489206, 0.49198192],
                [-0.41461205, 0.24869855, 0.25285226]]
                )
            
            if model_type == 'backbone':
                expected_tensor = backbone_base_expected_tensor if model_size == 'base' else backbone_large_expected_tensor
            print(outputs.last_hidden_state[0, :3, :3])
            assert torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_tensor, atol=1e-5), "Output does not match expected tensor."
            print("Inference successful, output matches expected tensor.") 
    


if __name__ == "__main__":
    convert(model_type='lvt', model_size='large', convert=True, upload=False, load_model=False, load_video=False, inference=False)
