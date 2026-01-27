import os
import re

import httpx
from huggingface_hub import hf_hub_download, snapshot_download

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
    "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png",
    "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/two_dogs.jpg",
    "https://llava-vl.github.io/static/images/view.jpg",
    "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
    "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/tiny_video.mp4",
    "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
    "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/tiny_video.mp4",
]


def url_to_local_path(url, return_url_if_not_found=True):
    filename = url.split("/")[-1]

    if not os.path.exists(filename) and return_url_if_not_found:
        return url

    return filename


def parse_hf_url(url):
    """
    Parse a HuggingFace Hub URL into components for hf_hub_download.

    Returns dict with (repo_id, filename, repo_type, revision) or None if not a HF URL.
    """
    pattern = r"https://huggingface\.co/(datasets/)?([^/]+/[^/]+)/resolve/([^/]+)/(.+)"
    match = re.match(pattern, url)
    if not match:
        return None

    is_dataset = match.group(1) is not None
    revision = match.group(3)
    return {
        "repo_id": match.group(2),
        "filename": match.group(4),
        "repo_type": "dataset" if is_dataset else "model",
        "revision": revision if revision != "main" else None,
    }


def validate_downloaded_content(filepath):
    with open(filepath, "r") as f:
        header = f.read(32)

    for bad_sig in ["<!doctype", "<html", '{"error', '{"message']:
        if header.lower().startswith(bad_sig):
            raise ValueError(
                f"Downloaded file appears to be an HTML error page, not a valid media file. "
                f"This may indicate rate limiting. File starts with: {header[:50]!r}"
            )

    file_size = os.path.getsize(filepath)
    if file_size < 100:
        raise ValueError(f"Downloaded file is suspiciously small ({file_size} bytes).")

    return True


def download_test_file(url):
    """
    Download a URL to a local file, using hf_hub_download for HF URLs.

    For HuggingFace URLs, uses hf_hub_download which handles authentication
    automatically via the HF_TOKEN environment variable.

    Returns the local filename.
    """
    filename = url.split("/")[-1]

    # Skip if file already exists
    if os.path.exists(filename):
        print(f"File already exists: {filename}")
        return filename

    # Check if this is a HuggingFace URL
    hf_parts = parse_hf_url(url)

    if hf_parts:
        # Use hf_hub_download for HF URLs - handles auth automatically via HF_TOKEN env var
        print(f"Downloading {filename} from HuggingFace Hub...")
        try:
            cached_path = hf_hub_download(
                repo_id=hf_parts["repo_id"],
                filename=hf_parts["filename"],
                repo_type=hf_parts["repo_type"],
                revision=hf_parts["revision"],
            )
            os.symlink(cached_path, filename)
            print(f"Successfully downloaded: {filename}")
        except Exception as e:
            print(f"Error downloading {filename} from HuggingFace Hub: {e}")
            raise
    else:
        # Use httpx for non-HF URLs (COCO, Britannica, etc.)
        print(f"Downloading {filename} from external URL...")
        try:
            with open(filename, "wb") as f:
                with httpx.stream("GET", url, follow_redirects=True) as resp:
                    resp.raise_for_status()
                    f.writelines(resp.iter_bytes(chunk_size=8192))

            # Validate the downloaded content
            validate_downloaded_content(filename)
            print(f"Successfully downloaded: {filename}")
        except httpx.HTTPError as e:
            print(f"Error downloading {filename}: {e}")
            raise
        except ValueError as e:
            # Validation failed - remove corrupted file
            print(f"Validation failed for {filename}: {e}")
            raise

    return filename


if __name__ == "__main__":
    if _run_pipeline_tests:
        import datasets

        _ = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        _ = datasets.load_dataset("hf-internal-testing/fixtures_image_utils", split="test", revision="refs/pr/1")
        _ = hf_hub_download(repo_id="nateraw/video-demo", filename="archery.mp4", repo_type="dataset")

        hf_hub_download("Narsil/asr_dummy", filename="hindi.ogg", repo_type="dataset")
        hf_hub_download(repo_id="hf-internal-testing/bool-masked-pos", filename="bool_masked_pos.pt")
        hf_hub_download(
            repo_id="hf-internal-testing/fixtures_docvqa",
            filename="nougat_pdf.png",
            repo_type="dataset",
            revision="ec57bf8c8b1653a209c13f6e9ee66b12df0fc2db",
        )
        hf_hub_download(
            repo_id="hf-internal-testing/image-matting-fixtures", filename="image.png", repo_type="dataset"
        )
        hf_hub_download(
            repo_id="hf-internal-testing/image-matting-fixtures", filename="trimap.png", repo_type="dataset"
        )
        hf_hub_download(
            repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti.npy", repo_type="dataset"
        )
        hf_hub_download(
            repo_id="hf-internal-testing/spaghetti-video",
            filename="eating_spaghetti_32_frames.npy",
            repo_type="dataset",
        )
        hf_hub_download(
            repo_id="hf-internal-testing/spaghetti-video",
            filename="eating_spaghetti_8_frames.npy",
            repo_type="dataset",
        )
        hf_hub_download(
            repo_id="hf-internal-testing/tourism-monthly-batch", filename="train-batch.pt", repo_type="dataset"
        )
        hf_hub_download(repo_id="huggyllama/llama-7b", filename="tokenizer.model")
        hf_hub_download(
            repo_id="nielsr/audio-spectogram-transformer-checkpoint", filename="sample_audio.flac", repo_type="dataset"
        )
        hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_pdf.png")
        hf_hub_download(
            repo_id="nielsr/test-image",
            filename="llava_1_6_input_ids.pt",
            repo_type="dataset",
        )
        hf_hub_download(
            repo_id="nielsr/test-image",
            filename="llava_1_6_pixel_values.pt",
            repo_type="dataset",
        )
        hf_hub_download(repo_id="nielsr/textvqa-sample", filename="bus.png", repo_type="dataset")
        hf_hub_download(
            repo_id="raushan-testing-hf/images_test",
            filename="emu3_image.npy",
            repo_type="dataset",
        )
        hf_hub_download(repo_id="raushan-testing-hf/images_test", filename="llava_v1_5_radar.jpg", repo_type="dataset")
        hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
        hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="video_demo.npy", repo_type="dataset")
        hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="video_demo_2.npy", repo_type="dataset")
        hf_hub_download(
            repo_id="shumingh/perception_lm_test_images",
            filename="14496_0.PNG",
            repo_type="dataset",
        )
        hf_hub_download(
            repo_id="shumingh/perception_lm_test_videos",
            filename="GUWR5TyiY-M_000012_000022.mp4",
            repo_type="dataset",
        )
        repo_id = "nielsr/image-segmentation-toy-data"
        hf_hub_download(
            repo_id="nielsr/image-segmentation-toy-data",
            filename="instance_segmentation_image_1.png",
            repo_type="dataset",
        )
        hf_hub_download(
            repo_id="nielsr/image-segmentation-toy-data",
            filename="instance_segmentation_image_2.png",
            repo_type="dataset",
        )
        hf_hub_download(
            repo_id="nielsr/image-segmentation-toy-data",
            filename="instance_segmentation_annotation_1.png",
            repo_type="dataset",
        )
        hf_hub_download(
            repo_id="nielsr/image-segmentation-toy-data",
            filename="instance_segmentation_annotation_2.png",
            repo_type="dataset",
        )
        hf_hub_download(
            repo_id="nielsr/image-segmentation-toy-data",
            filename="semantic_segmentation_annotation_1.png",
            repo_type="dataset",
        )
        hf_hub_download(
            repo_id="nielsr/image-segmentation-toy-data",
            filename="semantic_segmentation_annotation_2.png",
            repo_type="dataset",
        )
        hf_hub_download(
            repo_id="nielsr/image-segmentation-toy-data",
            filename="semantic_segmentation_image_1.png",
            repo_type="dataset",
        )
        hf_hub_download(
            repo_id="nielsr/image-segmentation-toy-data",
            filename="semantic_segmentation_image_2.png",
            repo_type="dataset",
        )
        hf_hub_download("shi-labs/oneformer_demo", "ade20k_panoptic.json", repo_type="dataset")

        hf_hub_download(
            repo_id="nielsr/audio-spectogram-transformer-checkpoint", filename="sample_audio.flac", repo_type="dataset"
        )

    # Need to specify the username on the endpoint `hub-ci`, otherwise we get
    # `fatal: could not read Username for 'https://hub-ci.huggingface.co': Success`
    # But this repo. is never used in a test decorated by `is_staging_test`.
    if not _run_staging:
        if not os.path.isdir("tiny-random-custom-architecture"):
            snapshot_download(
                "hf-internal-testing/tiny-random-custom-architecture",
                local_dir="tiny-random-custom-architecture",
            )

        # For `tests/test_tokenization_mistral_common.py:TestMistralCommonBackend`, which eventually calls
        # `mistral_common.tokens.tokenizers.utils.download_tokenizer_from_hf_hub` which (probably) doesn't have the cache.
        if is_mistral_common_available():
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

            from transformers import AutoTokenizer
            from transformers.tokenization_mistral_common import MistralCommonBackend

            repo_id = "hf-internal-testing/namespace-mistralai-repo_name-Mistral-Small-3.1-24B-Instruct-2503"
            AutoTokenizer.from_pretrained(repo_id, tokenizer_type="mistral")
            MistralCommonBackend.from_pretrained(repo_id)
            MistralTokenizer.from_hf_hub(repo_id)

            repo_id = "mistralai/Voxtral-Mini-3B-2507"
            AutoTokenizer.from_pretrained(repo_id)
            MistralTokenizer.from_hf_hub(repo_id)

    # Download files from URLs to local directory
    for url in URLS_FOR_TESTING_DATA:
        download_test_file(url)
