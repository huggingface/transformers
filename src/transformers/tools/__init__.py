from .agents import Agent, EndpointAgent, OpenAiAgent
from .base import PipelineTool, RemoteTool, tool
from .controlnet import ControlNetTool
from .generative_question_answering import GenerativeQuestionAnsweringTool, RemoteGenerativeQuestionAnsweringTool
from .image_captioning import ImageCaptioningTool, RemoteImageCaptioningTool
from .image_segmentation import ImageSegmentationTool
from .language_identifier import LanguageIdentificationTool
from .speech_to_text import RemoteSpeechToTextTool, SpeechToTextTool
from .stable_diffusion import StableDiffusionTool
from .text_classification import RemoteTextClassificationTool, TextClassificationTool
from .text_to_speech import TextToSpeechTool
from .translation import TranslationTool
