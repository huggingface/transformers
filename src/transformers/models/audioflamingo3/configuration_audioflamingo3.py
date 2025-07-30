from ... import PretrainedConfig

# Model Constants
IGNORE_INDEX = -100
SENTINEL_TOKEN = "<vila/sentinel>"

MEDIA_TOKENS = {
    "sound": "<sound>",
}

NUM_EXTRA_TOKENS = 10


class LlavaConfig(PretrainedConfig):
    model_type = "llava"

    def __init__(
        self,
        llm_cfg=None,
        sound_tower_cfg=None,
        sound_mm_projector_cfg=None,
        architectures=None,
        resume_path=None,
        hidden_size=None,
        sound_hidden_size=None,
        sound_encoder: str = '{"_target_": "llava.encoders.BasicSoundEncoder"}',
        **kwargs,
    ):
        super().__init__()
        self.architectures = architectures
        self.llm_cfg = llm_cfg
        self.sound_tower_cfg = sound_tower_cfg
        self.sound_mm_projector_cfg = sound_mm_projector_cfg
        self.resume_path = resume_path
        self.hidden_size = hidden_size
        self.sound_hidden_size = sound_hidden_size
        self.sound_encoder = '{"_target_": "llava.encoders.BasicSoundEncoder"}'


class SoundMultimodalProjectorConfig(PretrainedConfig):
    model_type = "sound_mm_projector"

    def __init__(self, sound_mm_projector_type: str = None, **kwargs):
        super().__init__()
        self.sound_mm_projector_type = sound_mm_projector_type
