from .bert import BertLayerFast


FAST_LAYERS_MAPPING_DICT = {"BertLayer": BertLayerFast, "ElectraLayer": BertLayerFast}


def is_module_fast(module_name):
    if module_name not in FAST_LAYERS_MAPPING_DICT.keys():
        return False
    else:
        return FAST_LAYERS_MAPPING_DICT[module_name]
