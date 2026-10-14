from transformers.models.llama.configuration_llama import LlamaConfig


class DuplicatedMethodConfig(LlamaConfig):
    @property
    def vocab_size(self):  # noqa: F811 -> we need this at we cannot delete the original for now since config dataclass refactor
        return 45

    @vocab_size.setter
    def vocab_size(self, value):
        self.vocab_size = value
