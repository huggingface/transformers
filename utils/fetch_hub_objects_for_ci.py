import os

import requests
from huggingface_hub import Repository, hf_hub_download

from transformers.testing_utils import _run_pipeline_tests, _run_staging


URLS_FOR_TESTING_DATA = [
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png",
    "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/Big_Buck_Bunny_720_10s_10MB.mp4",
    "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
    "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/glass-breaking-151256.mp3",
    "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/f2641_0_throatclearing.wav",
    "https://www.ilankelman.org/stopsigns/australia.jpg",
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
    "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
    "https://llava-vl.github.io/static/images/view.jpg",
    "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
]


def url_to_local_path(url, return_url_if_not_found=True):
    filename = url.split("/")[-1]

    if not os.path.exists(filename) and return_url_if_not_found:
        return url

    return filename


if __name__ == "__main__":
    if _run_pipeline_tests:
        import datasets

        _ = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        _ = datasets.load_dataset("hf-internal-testing/fixtures_image_utils", split="test", revision="refs/pr/1")
        _ = hf_hub_download(repo_id="nateraw/video-demo", filename="archery.mp4", repo_type="dataset")

    # Need to specify the username on the endpoint `hub-ci`, otherwise we get
    # `fatal: could not read Username for 'https://hub-ci.huggingface.co': Success`
    # But this repo. is never used in a test decorated by `is_staging_test`.
    if not _run_staging:
        # Used in as `tests/models/auto/test_modeling_auto.py::AutoModelTest::test_dynamic_saving_from_local_repo --> _ = Repository( ... )`
        # TODO: Remove this and the above test when `huggingface_hub v1.0` comes (where `Repository` will be removed).
        _ = Repository(
            local_dir="tiny-random-custom-architecture",
            clone_from="hf-internal-testing/tiny-random-custom-architecture",
        )

    # Download files from URLs to local directory
    for url in URLS_FOR_TESTING_DATA:
        filename = url_to_local_path(url, return_url_if_not_found=False)

        # Skip if file already exists
        if os.path.exists(filename):
            print(f"File already exists: {filename}")
            continue

        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Successfully downloaded: {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filename}: {e}")
