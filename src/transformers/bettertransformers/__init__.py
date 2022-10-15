# from .bert import BertLayerFast
from . import bert


FAST_LAYERS_MAPPING_DICT = {"BertLayer": bert.BertLayerFast, "ElectraLayer": bert.BertLayerFast}


def is_module_fast(module_name):
    if module_name not in FAST_LAYERS_MAPPING_DICT.keys():
        return False
    else:
        return FAST_LAYERS_MAPPING_DICT[module_name]
