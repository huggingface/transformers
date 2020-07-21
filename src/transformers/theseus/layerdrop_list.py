from .theseus_list import TheseusList
from .theseus_module import TheseusModule


class LayerDropList(TheseusList):
    """
    Implementation of Layer Drop (https://arxiv.org/abs/1909.11556).
    """

    @classmethod
    def from_module_list(cls, module_list, replacing_rate):
        list_to_return = cls()
        for module in module_list:
            list_to_return.append(TheseusModule(successor=module, replacing_rate=replacing_rate))
        return list_to_return
