from transformers.models.llama.modeling_llama import LlamaConfig
# Example where we only want to only add a new config argument and new arg doc
# here there is no `ARG` so we are gonna take parent doc
class MyNewModelConfig(LlamaConfig):
    r"""
        mlp_bias (`bool`, *optional*, defaults to `False`)
    """
    def __init__(
        self,
        mlp_bias=False 
    ):
        self.mlp_bias = mlp_bias
        super().__init__(self)