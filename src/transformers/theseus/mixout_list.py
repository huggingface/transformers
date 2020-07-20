from copy import deepcopy
from .theseus_list import TheseusList
from .theseus_module import TheseusModule


class MixoutList(TheseusList):
    """
    Implementation of Mixout (https://arxiv.org/abs/1909.11299).
    """
    @classmethod
    def from_module_list(cls, module_list, replacing_rate):
        list_to_return = cls()
        for module in module_list:
            list_to_return.append(
                TheseusModule(predecessor=module,
                              successor=deepcopy(module),
                              replacing_rate=replacing_rate)
            )
        return list_to_return
