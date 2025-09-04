import os

import requests
from huggingface_hub import Repository, hf_hub_download

from transformers.testing_utils import _run_pipeline_tests, _run_staging
from transformers.utils.import_utils import is_mistral_common_available


URLS_FOR_TESTING_DATA = [
    "http://images.cocodataset.org/val2017/000000000139.jpg",
    "http://images.cocodataset.org/val2017/000000000285.jpg",
    "http://images.cocodataset.org/val2017/000000000632.jpg",
    "http://images.cocodataset.org/val2017/000000000724.jpg",
    "http://images.cocodataset.org/val2017/000000000776.jpg",
    "http://images.cocodataset.org/val2017/000000000785.jpg",
    "http://images.cocodataset.org/val2017/000000000802.jpg",
    "http://images.cocodataset.org/val2017/000000000872.jpg",
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg",
    "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
    "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bus.png",
    "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg",
    "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/f2641_0_throatclearing.wav",
    "https://huggingface.co/datasets/raushan-testing-hf/audio-test/resolve/main/glass-breaking-151256.mp3",
    "https://huggingface.co/datasets/raushan-testing-hf/images_test/resolve/main/picsum_237_200x300.jpg",
    "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/Big_Buck_Bunny_720_10s_10MB.mp4",
    "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4",
    "https://huggingface.co/kirp/kosmos2_5/resolve/main/receipt_00008.png",
    "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/two_dogs.jpg",
    "https://llava-vl.github.io/static/images/view.jpg",
    "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
    "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/tiny_video.mp4",
    "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
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

        # For `tests/test_tokenization_mistral_common.py:TestMistralCommonTokenizer`, which eventually calls
        # `mistral_common.tokens.tokenizers.utils.download_tokenizer_from_hf_hub` which (probably) doesn't have the cache.
        if is_mistral_common_available():
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

            from transformers import AutoTokenizer
            from transformers.tokenization_mistral_common import MistralCommonTokenizer

            repo_id = "hf-internal-testing/namespace-mistralai-repo_name-Mistral-Small-3.1-24B-Instruct-2503"
            AutoTokenizer.from_pretrained(repo_id, tokenizer_type="mistral")
            MistralCommonTokenizer.from_pretrained(repo_id)
            MistralTokenizer.from_hf_hub(repo_id)

            repo_id = "mistralai/Voxtral-Mini-3B-2507"
            AutoTokenizer.from_pretrained(repo_id)
            MistralTokenizer.from_hf_hub(repo_id)

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
