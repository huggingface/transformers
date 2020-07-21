from copy import deepcopy

from .theseus_list import TheseusList
from .theseus_module import TheseusModule


class MixoutList(TheseusList):
    """
    Implementation of Mixout (https://arxiv.org/abs/1909.11299).
    """

    @classmethod
    def from_module_list(cls, module_list, replacing_rate, freeze_predecessor=True):
        """
        :param module_list:
        :param replacing_rate:
        :param freeze_predecessor: whether to freeze the original pretraining weights.
        :return:
        """
        list_to_return = cls()
        for module in module_list:
            predecessor = deepcopy(module)
            if freeze_predecessor:
                for param in predecessor.parameters():
                    param.requires_grad = False
            list_to_return.append(
                TheseusModule(predecessor=predecessor, successor=module, replacing_rate=replacing_rate)
            )
        return list_to_return
