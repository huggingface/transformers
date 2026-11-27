from transformers import CLIPImageProcessor


class CustomImageProcessor(CLIPImageProcessor):
    pass


class CustomImageProcessorOwnFromDict(CLIPImageProcessor):
    # Like some remote-code image processors (e.g. moonshotai/Kimi-K3): `from_dict` overridden without
    # `return_unused_kwargs` support, so it always returns the bare object.
    @classmethod
    def from_dict(cls, image_processor_dict, **kwargs):
        kwargs.pop("return_unused_kwargs", None)
        return cls(**{**image_processor_dict, **kwargs})
