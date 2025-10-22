from transformers import AutoProcessor, Glm4vForConditionalGeneration
from PIL import Image
import cv2


def prepare_video_metadata(videos):
    video_metadata = []
    for video in videos:
        if isinstance(video, list):
            num_frames = len(video)
        elif hasattr(video, "shape"):
            if len(video.shape) == 4:  # (T, H, W, C)
                num_frames = video.shape[0]
            else:
                num_frames = 1
        else:
            num_frames = 8
            print("eeeeee")

        metadata = {
            "fps": 2,
            "duration": num_frames / 2,
            "total_frames": num_frames,
        }
        video_metadata.append(metadata)
    return video_metadata


def test_video_processing(video_path_list, num_frames=300):
    selected_frames = []
    for video_path in video_path_list:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {frame_count}")

    video_metadata = []
    for video_path in video_path_list:
        temp_frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(frame_count // num_frames, 1)
        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            temp_frames.append(pil_img)
        selected_frames.append(temp_frames)

    video_metadata = prepare_video_metadata(selected_frames)
    processor = AutoProcessor.from_pretrained("/model/glm-4v-9b-0529", use_fast=True)
    video_inputs = processor.video_processor(
        videos=selected_frames, video_metadata=video_metadata, return_tensors="pt"
    )
    breakpoint()


if __name__ == "__main__":
    video_path_1 = "/workspace/mmdoctor/datasets_platform/Video/MMVU/medias/Civil_Engineering_0.mp4"
    video_path_2 = "/workspace/mmdoctor/datasets_platform/Video/MMVU/medias/Civil_Engineering_1.mp4"
    test_video_processing([video_path_1, video_path_2])
