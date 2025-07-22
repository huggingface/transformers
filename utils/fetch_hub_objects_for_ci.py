from huggingface_hub import hf_hub_download

from transformers.testing_utils import _run_pipeline_tests


if __name__ == "__main__":
    if _run_pipeline_tests:
        import datasets

        _ = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        _ = datasets.load_dataset("hf-internal-testing/fixtures_image_utils", split="test", revision="refs/pr/1")
        _ = hf_hub_download(repo_id="nateraw/video-demo", filename="archery.mp4", repo_type="dataset")
