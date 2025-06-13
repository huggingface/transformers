from huggingface_hub import HfApi
from pathlib import Path

# --- Configuration ---
# Your Hugging Face username
username = "shumingh"

# Local paths to your files
image_file_path = "/home/smhu/code/occhi/apps/plm/dummy_datasets/image/images/14496_0.PNG"
video_file_path = "/home/smhu/code/occhi/apps/plm/dummy_datasets/video/videos/GUWR5TyiY-M_000012_000022.mp4"

# Desired repository IDs on the Hub
image_repo_id = f"{username}/perception_lm_test_images"
video_repo_id = f"{username}/perception_lm_test_videos"
# --- End of Configuration ---

# Initialize the API
api = HfApi()

# Create the dataset repositories (if they don't exist)
api.create_repo(repo_id=image_repo_id, repo_type="dataset", exist_ok=True)
api.create_repo(repo_id=video_repo_id, repo_type="dataset", exist_ok=True)

print(f"Uploading {image_file_path} to {image_repo_id}...")
api.upload_file(
    path_or_fileobj=image_file_path,
    path_in_repo=Path(image_file_path).name,
    repo_id=image_repo_id,
    repo_type="dataset",
)
print("Image upload complete.")

print(f"Uploading {video_file_path} to {video_repo_id}...")
api.upload_file(
    path_or_fileobj=video_file_path,
    path_in_repo=Path(video_file_path).name,
    repo_id=video_repo_id,
    repo_type="dataset",
)
print("Video upload complete.")

print("\n--- URLs ---")
print(f"Image: https://huggingface.co/datasets/{image_repo_id}/blob/main/{Path(image_file_path).name}")
print(f"Video: https://huggingface.co/datasets/{video_repo_id}/blob/main/{Path(video_file_path).name}") 