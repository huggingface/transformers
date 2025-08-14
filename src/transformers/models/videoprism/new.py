# from transformers.video_utils import load_video
# from transformers.models.videoprism.video_processing_videoprism import VideoPrismVideoProcessor
# from huggingface_hub import hf_hub_download
# import numpy as np
# import mediapy
# import torch



# def prepare_video():
#     file = hf_hub_download(
#         repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti_32_frames.npy", repo_type="dataset"
#     )
#     video = np.load(file)
#     return list(video)

# # inputs = load_video(
# #     "./src/transformers/models/videoprism/water_bottle_drumming.mp4",
# # )

# # print(inputs[0].shape, inputs[1])

# def read_and_preprocess_video(   # This function from the original code
#     filename: str, target_num_frames: int, target_frame_size: tuple[int, int]
#     ):
#     """Reads and preprocesses a video."""
#     try:
#         frames = mediapy.read_video(filename)
#     except:
#         frames = prepare_video()
#         print("done")
        
#     # Sample to target number of frames.
#     frame_indices = np.linspace(
#         0, len(frames), num=target_num_frames, endpoint=False, dtype=np.int32
#     )
#     frames = np.array([frames[i] for i in frame_indices])

#     # Resize to target size.
#     # original_height, original_width = frames.shape[-3:-1]
#     # target_height, target_width = target_frame_size
#     # assert (
#     #     original_height * target_width == original_width * target_height
#     # ), 'Currently does not support aspect ratio mismatch.'
#     frames = mediapy.resize_video(frames, shape=target_frame_size)

#     # Normalize pixel values to [0.0, 1.0].
#     frames = mediapy.to_float01(frames)

#     return frames






# def compare_inputs():
#     # get the spaghetti video
#     # load it via old processing function
#     frames = read_and_preprocess_video(None, 16, (288, 288))

#     # convert to torch and name it old_inputs
#     old_inputs = torch.tensor(frames).unsqueeze(0).permute(0, 1, 4, 2, 3)   #? (1, 16, 3, 288, 288)
    
#     # load the spaghetti video via the video processor function
#     inputs = prepare_video()
#     new_inputs = VideoPrismVideoProcessor()(inputs, return_tensors="pt")
#     # print the outputs
#     print(f"{old_inputs.shape=}, {new_inputs['pixel_values_videos'].shape=}")
#     # assert the values
    
#     return old_inputs, new_inputs['pixel_values_videos']


# if __name__ == "main":
#     print("all good here")
#     old, new = compare_inputs()
#     print(old.shape)
#     print(new.shape)
    
    # Example usage
    # video_path = "./src/transformers/models/videoprism/water_bottle_drumming.mp4"
    # target_num_frames = 16
    # target_frame_size = (288, 288)
    
    # frames = read_and_preprocess_video(video_path, target_num_frames, target_frame_size)
    # print(frames.shape)  # Should print the shape of the processed frames


from transformers.video_utils import load_video
from transformers.models.videoprism.video_processing_videoprism import VideoPrismVideoProcessor
from huggingface_hub import hf_hub_download
import numpy as np
import mediapy
import torch
from transformers import VivitConfig, VivitForVideoClassification, VivitImageProcessor


def prepare_video():
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti_32_frames.npy", repo_type="dataset"
    )
    video = np.load(file)
    return list(video)

# inputs = load_video(
#     "./src/transformers/models/videoprism/water_bottle_drumming.mp4",
# )

# print(inputs[0].shape, inputs[1])

def read_and_preprocess_video(   # This function from the original code
    filename: str, target_num_frames: int, target_frame_size: tuple[int, int]
    ):
    """Reads and preprocesses a video."""
    try:
        frames = mediapy.read_video(filename)
    except:
        frames = prepare_video()
        print("done")
        
    # Sample to target number of frames.
    frame_indices = np.linspace(
        0, len(frames), num=target_num_frames, endpoint=False, dtype=np.int32
    )
    frames = np.array([frames[i] for i in frame_indices])

    # Resize to target size.
    # original_height, original_width = frames.shape[-3:-1]
    # target_height, target_width = target_frame_size
    # assert (
    #     original_height * target_width == original_width * target_height
    # ), 'Currently does not support aspect ratio mismatch.'
    frames = mediapy.resize_video(frames, shape=target_frame_size)

    # Normalize pixel values to [0.0, 1.0].
    frames = mediapy.to_float01(frames)

    return frames






def compare_inputs():
    # get the spaghetti video
    # load it via old processing function
    frames = read_and_preprocess_video(None, 16, (288, 288))

    # convert to torch and name it old_inputs
    old_inputs = torch.tensor(frames).unsqueeze(0).permute(0, 1, 4, 2, 3)   #? (1, 16, 3, 288, 288)
    
    # load the spaghetti video via the video processor function
    inputs = prepare_video()
    frame_indices = np.linspace(
        0, len(inputs), num=16, endpoint=False, dtype=np.int32
    )
    inputs = np.array([inputs[i] for i in frame_indices])
    new_inputs = VivitImageProcessor()(inputs, return_tensors="pt")
    # print the outputs
    print(f"{old_inputs.shape=}, {new_inputs['pixel_values_videos'].shape=}")
    # assert the values
    
    return old_inputs, new_inputs['pixel_values_videos']

print("all good here")
old, new = compare_inputs()
print(old[0,0,0,:3,:3])
print(new[0,0,0,:3,:3])