from typing import Any, TypeVar



class PretrainedConfig:
    model_type: str
    num_labels: int | None = None  

    def __init__(self, **kwargs: Any):

        for k, v in kwargs.items():
            setattr(self, k, v)


        if hasattr(self, "num_labels") and self.num_labels is not None:
            if not isinstance(self.num_labels, int):
                raise TypeError(f"num_labels must be int, got {type(self.num_labels)}")


    @classmethod
    def from_pretrained(
        cls: TypeVar("PretrainedConfig"), *args, **kwargs
    ) -> "PretrainedConfig":
        config = cls(**kwargs)
        # Type check example
        if config.num_labels is not None and not isinstance(config.num_labels, int):
            raise TypeError(f"num_labels must be int, got {type(config.num_labels)}")
        return config

    # Ensure type hints for to_dict
    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            result[k] = v
        return result


if __name__ == "__main__":
    # Correct type
    cfg = PretrainedConfig(num_labels=5)
    print("Correct type test passed:", cfg.num_labels)


    try:
        cfg_bad = PretrainedConfig(num_labels="five")
    except TypeError as e:
        print("Incorrect type test passed:", e)
