import tempfile

import numpy as np
import PIL.Image
import soundfile as sf
import torch
from PIL import Image
from PIL.Image import Image as ImageType


class AgentType:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.to_string()

    def to_raw(self):
        raise NotImplementedError

    def to_string(self) -> str:
        raise NotImplementedError


class AgentText(AgentType, str):
    def to_raw(self):
        return self.value

    def to_string(self):
        return self.value


class AgentImage(AgentType, PIL.Image.Image):
    def __init__(self, value):
        super().__init__(value)

        self._path = None
        self._raw = None
        self._tensor = None

        if isinstance(value, ImageType):
            self._raw = value
        elif isinstance(value, str):
            self._path = value
        elif isinstance(value, torch.Tensor):
            self._tensor = value
        else:
            raise ValueError(f"Unsupported type for {self.__class__.__name__}: {type(value)}")

    def _ipython_display_(self, include=None, exclude=None):
        """
        Displays correctly this type in an ipython notebook (ipython, colab, jupyter, ...)
        """
        from IPython.display import Image, display

        display(Image(self.to_string()))

    def to_raw(self):
        """
        Returns the "raw" version of that object. In the case of an AgentImage, it is a PIL.Image.
        """
        if self._raw is not None:
            return self._raw

        if self._path is not None:
            return Image.open(self._path)

    def to_string(self):
        """
        Returns the stringified version of that object. In the case of an AgentImage, it is a path to the serialized
        version of the image.
        """
        if self._path is not None:
            return self._path

        if self._raw is not None:
            temp = tempfile.NamedTemporaryFile(suffix=".png")
            self._path = temp.name
            self._raw.save(self._path)
            temp.close()

            return self._path

        if self._tensor is not None:
            array = self._tensor.cpu().detach().numpy()
            array[array <= 0] = 0
            array[array > 0] = 1

            # There is likely simpler than load into image into save
            img = Image.fromarray((array * 255).astype(np.uint8))
            temp = tempfile.NamedTemporaryFile(suffix=".png")
            img.save(temp.name)

            temp.close()


class AgentAudio(AgentType):
    def __init__(self, value, samplerate=16000):
        super().__init__(value)

        self._path = None
        self._tensor = None

        self.samplerate = samplerate

        if isinstance(value, str):
            self._path = value
        elif isinstance(value, torch.Tensor):
            self._tensor = value
        else:
            raise ValueError(f"Unsupported audio type: {type(value)}")

    def _ipython_display_(self, include=None, exclude=None):
        from IPython.display import Audio, display

        display(Audio(self.to_string(), rate=self.samplerate))

    def to_raw(self):
        if self._tensor is not None:
            return self._tensor

        if self._path is not None:
            return sf.read(self._path, samplerate=self.samplerate)

    def to_string(self):
        if self._path is not None:
            return self._path

        if self._tensor is not None:
            temp = tempfile.NamedTemporaryFile(suffix=".wav")
            self._path = temp.name

            sf.write(self._path, self._tensor, samplerate=self.samplerate)

            temp.close()

            return self._path


class AgentVideo(AgentType):
    def __init__(self, value):
        super().__init__(value)

        self._path = None
        self._tensor = None

        if isinstance(value, str):
            self._path = value
        elif isinstance(value, torch.Tensor):
            self._tensor = value
        else:
            raise ValueError(f"Unsupported audio type: {type(value)}")

    def _ipython_display_(self, include=None, exclude=None):
        from IPython.display import Video, display

        display(Video(filename=self.to_string(), embed=True))

    def to_raw(self):
        if self._tensor is not None:
            return self._tensor

        if self._path is not None:
            import torchvision

            self._tensor = torchvision.io.read_video(self._path)[0]
            return self._tensor

    def to_string(self):
        if self._path is not None:
            return self._path

        if self._tensor is not None:
            import cv2

            temp = tempfile.NamedTemporaryFile(suffix=".mp4")
            self._path = temp.name

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_frames = self._tensor.numpy()

            h, w, c = video_frames[0].shape
            video_writer = cv2.VideoWriter(self._path, fourcc, fps=8, frameSize=(w, h))

            for i in range(len(video_frames)):
                img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
                video_writer.write(img)

            temp.close()
            return self._path


AGENT_TYPE_MAPPING = {"text": AgentText, "image": AgentImage, "audio": AgentAudio, "video": AgentVideo}
