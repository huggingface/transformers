
import unittest
from transformers.testing_utils import (
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers import PEAudioVideoWithTextModel, PEAudioVideoProcessor
import torch
from huggingface_hub import hf_hub_download

@require_vision
@require_torch
class PEAudioVideoWithTextModelIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the model and processor for use in tests
        cls.model = PEAudioVideoWithTextModel.from_pretrained("facebook/pe-av-large").to(torch_device)
        cls.processor = PEAudioVideoProcessor.from_pretrained("facebook/pe-av-large")
        cls.dot_products = {
            "audio_video": 0.44389599561691284,
            "audio_text": 0.37399545311927795,
            "video_text": 0.13941769301891327,
            "audio_video_text": 0.39156609773635864,
        }

    @slow
    def test_text_inference(self):
        inputs = self.processor(text=["A woman and a man speaking"], return_tensors="pt", padding=True).to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        self.assertIsNone(outputs.audio_embeds)
        self.assertIsNone(outputs.video_embeds)
        self.assertIsNone(outputs.audio_video_embeds)
        self.assertIsNone(outputs.audio_text_embeds)
        self.assertIsNone(outputs.video_text_embeds)
        self.assertEqual(outputs.audio_video_text_embeds.shape, torch.Size([1, 1024]))

    @slow
    def test_audio_inference(self):
        audio_path = hf_hub_download(
            repo_id="facebook/pe-av-test-videos",
            filename="audiobox.mp4",
            repo_type="dataset"
        )
        inputs = self.processor(audio=[audio_path], return_tensors="pt", padding=True).to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        self.assertIsNone(outputs.video_embeds)
        self.assertIsNone(outputs.audio_video_embeds)
        self.assertIsNone(outputs.audio_text_embeds)
        self.assertIsNone(outputs.video_text_embeds)
        self.assertIsNone(outputs.audio_video_text_embeds)
        self.assertEqual(outputs.audio_embeds.shape, torch.Size([1, 1024]))

    @slow
    def test_video_inference(self):
        video_path = hf_hub_download(
            repo_id="facebook/pe-av-test-videos",
            filename="audiobox.mp4",
            repo_type="dataset"
        )
        inputs = self.processor(videos=[video_path], return_tensors="pt", padding=True).to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        self.assertIsNone(outputs.audio_video_embeds)
        self.assertIsNone(outputs.audio_text_embeds)
        self.assertIsNone(outputs.video_text_embeds)
        self.assertIsNone(outputs.audio_video_text_embeds)
        self.assertIsNone(outputs.audio_embeds)
        self.assertEqual(outputs.video_embeds.shape, torch.Size([1, 1024]))

    @slow
    def test_audio_video_inference(self):
        video_path = hf_hub_download(
            repo_id="facebook/pe-av-test-videos",
            filename="audiobox.mp4",
            repo_type="dataset"
        )
        audio_path = hf_hub_download(
            repo_id="facebook/pe-av-test-videos",
            filename="audiobox.mp4",
            repo_type="dataset"
        )
        inputs = self.processor(videos=[video_path], audio=[audio_path], return_tensors="pt", padding=True).to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        self.assertIsNone(outputs.audio_text_embeds)
        self.assertIsNone(outputs.video_text_embeds)
        self.assertIsNone(outputs.audio_video_text_embeds)
        self.assertEqual(outputs.audio_embeds.shape, torch.Size([1, 1024]))
        self.assertEqual(outputs.video_embeds.shape, torch.Size([1, 1024]))
        self.assertEqual(outputs.audio_video_embeds.shape, torch.Size([1, 1024]))
        dotp = outputs.audio_embeds @ outputs.video_embeds.T
        self.assertAlmostEqual(dotp.item(), self.dot_products["audio_video"], delta=.01)

    @slow
    def test_audio_text_inference(self):
        audio_path = hf_hub_download(
            repo_id="facebook/pe-av-test-videos",
            filename="audiobox.mp4",
            repo_type="dataset"
        )
        inputs = self.processor(text=["A woman and a man speaking"], audio=[audio_path], return_tensors="pt", padding=True).to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        self.assertIsNone(outputs.audio_video_embeds)
        self.assertIsNone(outputs.video_embeds)
        self.assertIsNone(outputs.video_text_embeds)
        self.assertIsNone(outputs.audio_video_text_embeds)
        self.assertEqual(outputs.audio_embeds.shape, torch.Size([1, 1024]))
        self.assertEqual(outputs.audio_text_embeds.shape, torch.Size([1, 1024]))
        dotp = outputs.audio_embeds @ outputs.audio_text_embeds.T
        self.assertAlmostEqual(dotp.item(), self.dot_products["audio_text"], delta=.01)

    @slow
    def test_video_text_inference(self):
        video_path = hf_hub_download(
            repo_id="facebook/pe-av-test-videos",
            filename="audiobox.mp4",
            repo_type="dataset"
        )
        inputs = self.processor(text=["A woman and a man speaking"], videos=[video_path], return_tensors="pt", padding=True).to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        self.assertIsNone(outputs.audio_video_embeds)
        self.assertIsNone(outputs.audio_embeds)
        self.assertIsNone(outputs.audio_text_embeds)
        self.assertIsNone(outputs.audio_video_text_embeds)
        self.assertEqual(outputs.video_embeds.shape, torch.Size([1, 1024]))
        self.assertEqual(outputs.video_text_embeds.shape, torch.Size([1, 1024]))
        dotp = outputs.video_embeds @ outputs.video_text_embeds.T
        self.assertAlmostEqual(dotp.item(), self.dot_products["video_text"], delta=.01)

    @slow
    def test_audio_video_text_inference(self):
        video_path = hf_hub_download(
            repo_id="facebook/pe-av-test-videos",
            filename="audiobox.mp4",
            repo_type="dataset"
        )
        audio_path = hf_hub_download(
            repo_id="facebook/pe-av-test-videos",
            filename="audiobox.mp4",
            repo_type="dataset"
        )
        inputs = self.processor(text=["A woman and a man speaking"], audio=[audio_path], videos=[video_path], return_tensors="pt", padding=True).to(torch_device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        self.assertEqual(outputs.audio_embeds.shape, torch.Size([1, 1024]))
        self.assertEqual(outputs.video_embeds.shape, torch.Size([1, 1024]))
        self.assertEqual(outputs.video_embeds.shape, torch.Size([1, 1024]))
        self.assertEqual(outputs.video_embeds.shape, torch.Size([1, 1024]))
        self.assertEqual(outputs.video_embeds.shape, torch.Size([1, 1024]))
        self.assertEqual(outputs.video_embeds.shape, torch.Size([1, 1024]))
        av_dotp = outputs.audio_embeds @ outputs.video_embeds.T
        at_dotp = outputs.audio_embeds @ outputs.audio_text_embeds.T
        vt_dotp = outputs.video_embeds @ outputs.video_text_embeds.T
        avt_dotp = outputs.audio_video_embeds @ outputs.audio_video_text_embeds.T
        self.assertAlmostEqual(av_dotp.item(), self.dot_products["audio_video"], delta=.01)
        self.assertAlmostEqual(at_dotp.item(), self.dot_products["audio_text"], delta=.01)
        self.assertAlmostEqual(vt_dotp.item(), self.dot_products["video_text"], delta=.01)
        self.assertAlmostEqual(avt_dotp.item(), self.dot_products["audio_video_text"], delta=.01)
