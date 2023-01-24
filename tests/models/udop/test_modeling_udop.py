# from transformers.models.UDOP.modeling_udop_dual import UdopDualForConditionalGeneration
from transformers.models.UDOP.modeling_udop_uni import UdopUnimodelForConditionalGeneration
# from transformers.models.UDOP.configuration_udop import UdopConfig

udopunimodel = UdopUnimodelForConditionalGeneration.from_pretrained("/Users/eaxxkra/Downloads/UdopUnimodel-Large-224/",local_files_only=True)