# from transformers.models.UDOP.modeling_udop_dual import UdopDualForConditionalGeneration
from transformers.models.UDOP.modeling_udop_uni import UdopUnimodelForConditionalGeneration
from transformers.models.UDOP.modeling_udop_dual import UdopDualForConditionalGeneration
from transformers.models.UDOP.configuration_udop import UdopConfig
from transformers.models.UDOP.tokenization_udop import UdopTokenizer


def test():
    config = UdopConfig.from_pretrained("/Users/eaxxkra/Downloads/udop-dual-large-224")
    tokenizer = UdopTokenizer.from_pretrained("/Users/eaxxkra/Downloads/udop-dual-large-224")
    model = UdopDualForConditionalGeneration.from_pretrained("/Users/eaxxkra/Downloads/udop-dual-large-224")

    # config = UdopConfig.from_pretrained("/Users/eaxxkra/Downloads/udop-unimodel-large-224")
    # tokenizer = UdopTokenizer.from_pretrained("/Users/eaxxkra/Downloads/udop-unimodel-large-224")
    # model = UdopUnimodelForConditionalGeneration.from_pretrained("/Users/eaxxkra/Downloads/udop-unimodel-large-224")

    print(model)
# udopunimodel = UdopUnimodelForConditionalGeneration.from_pretrained("/Users/eaxxkra/Downloads/UdopUnimodel-Large-224/",local_files_only=True)