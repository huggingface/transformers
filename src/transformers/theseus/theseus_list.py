import torch

from .theseus_module import TheseusModule


def _unpack_module(packed_module):
    list_to_return = torch.nn.ModuleList()
    if isinstance(packed_module, (list, tuple, torch.nn.ModuleList)):
        for submodule in packed_module:
            list_to_return.append(submodule)
    elif isinstance(packed_module, torch.nn.Module):
        list_to_return.append(packed_module)
    return list_to_return


class TheseusList(torch.nn.ModuleList):
    """
        TheseusList is a ModuleList that implements methods for Theseus Compression.
    """

    def set_replacing_rate(self, replacing_rate):
        for module in self:
            if isinstance(module, TheseusModule):
                module.set_replacing_rate(replacing_rate)

    def sample_and_pass(self) -> torch.nn.ModuleList:
        list_to_return = torch.nn.ModuleList()
        for module in self:
            if isinstance(module, TheseusModule):
                list_to_return += _unpack_module(module.sample_and_pass())
            else:
                list_to_return += _unpack_module(module)
        return list_to_return

    def get_successors(self) -> torch.nn.ModuleList:
        list_to_return = torch.nn.ModuleList()
        for module in self:
            if isinstance(module, TheseusModule) and module.successor:
                list_to_return += _unpack_module(module.successor)
        return list_to_return

    def get_predecessors(self) -> torch.nn.ModuleList:
        list_to_return = torch.nn.ModuleList()
        for module in self:
            if isinstance(module, TheseusModule):
                list_to_return += _unpack_module(module.predecessor)
        return list_to_return
