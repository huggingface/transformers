from __future__ import annotations
from pprint import pprint

from transformers import CvtConfig, CvtForImageClassification
from lib.models.cls_cvt import ConvolutionalVisionTransformer as CVT
import torch
import yaml

import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass, field
from typing import List


@dataclass
class Tracker:
    """This class tracks all the operations of a given module by performing a forward pass.
    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from glasses.utils import Tracker
        >>> model = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64,10), nn.ReLU())
        >>> tr = Tracker(model)
        >>> tr(x)
        >>> print(tr.traced) # all operations
        >>> print('-----')
        >>> print(tr.parametrized) # all operations with learnable params
        outputs
        ``[Linear(in_features=1, out_features=64, bias=True),
        ReLU(),
        Linear(in_features=64, out_features=10, bias=True),
        ReLU()]
        -----
        [Linear(in_features=1, out_features=64, bias=True),
        Linear(in_features=64, out_features=10, bias=True)]``
    """

    module: nn.Module
    traced: List[nn.Module] = field(default_factory=list)
    handles: list = field(default_factory=list)

    def _forward_hook(self, m, inputs: Tensor, outputs: Tensor):
        has_not_submodules = (
            len(list(m.modules())) == 1
            or isinstance(m, nn.Conv2d)
            or isinstance(m, nn.BatchNorm2d)
        )
        if has_not_submodules:
            self.traced.append(m)

    def __call__(self, x: Tensor) -> Tracker:
        for m in self.module.modules():
            self.handles.append(m.register_forward_hook(self._forward_hook))
        self.module(x)
        list(map(lambda x: x.remove(), self.handles))
        return self

    @property
    def parametrized(self):
        # check the len of the state_dict keys to see if we have learnable params
        return list(filter(lambda x: len(list(x.state_dict().keys())) > 0, self.traced))

@dataclass
class ModuleTransfer:
    """This class transfers the weight from one module to another assuming
    they have the same set of operations but they were defined in a different way.
    :Examples
        >>> import torch
        >>> import torch.nn as nn
        >>> from eyes.utils import ModuleTransfer
        >>> model_a = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64,10), nn.ReLU())
        >>> def block(in_features, out_features):
        >>>     return nn.Sequential(nn.Linear(in_features, out_features),
                                nn.ReLU())
        >>> model_b = nn.Sequential(block(1,64), block(64,10))
        >>> # model_a and model_b are the same thing but defined in two different ways
        >>> x = torch.ones(1, 1)
        >>> trans = ModuleTransfer(src=model_a, dest=model_b)
        >>> trans(x)
        # now module_b has the same weight of model_a
    """

    src: nn.Module
    dest: nn.Module
    verbose: int = 0
    src_skip: List = field(default_factory=list)
    dest_skip: List = field(default_factory=list)

    def __call__(self, x: Tensor):
        """Transfer the weights of `self.src` to `self.dest` by performing a forward pass using `x` as input.
        Under the hood we tracked all the operations in booth modules.
        :param x: [The input to the modules]
        :type x: torch.tensor
        """
        dest_traced = Tracker(self.dest)(x).parametrized
        src_traced = Tracker(self.src)(x).parametrized

        src_traced = list(filter(lambda x: type(x) not in self.src_skip, src_traced))
        dest_traced = list(filter(lambda x: type(x) not in self.dest_skip, dest_traced))

        if len(dest_traced) != len(src_traced):
            raise Exception(
                f"Numbers of operations are different. Source module has {len(src_traced)} operations while destination module has {len(dest_traced)}."
            )

        for dest_m, src_m in zip(dest_traced, src_traced):
            dest_m.load_state_dict(src_m.state_dict())
            if self.verbose == 1:
                print(f"Transfered from={src_m} to={dest_m}")
# Repo link: https://github.com/microsoft/CvT
# Model Zoo: https://1drv.ms/u/s!AhIXJn_J-blW9RzF3rMW7SsLHa8h?e=blQ0Al
# yaml is in experiments folder

with open('C:\\Users\AH87766\Documents\CvT\experiments\imagenet\cvt\cvt-13-224x224.yaml', 'r') as f:
    original_config = yaml.load(f, Loader=yaml.FullLoader)

cvt_hugging_config = CvtConfig()
cvt_hugging_config.cls_token =  [False, False, True]
cvt_hugging_model = CvtForImageClassification(cvt_hugging_config).eval()
original_config['MODEL']['CLS_TOKEN'] = [False, False, True]
original_model = CVT(spec=original_config['MODEL']['SPEC']).eval()

x = torch.zeros((1, 3, 224, 224))
module_transfer = ModuleTransfer(original_model, cvt_hugging_model,verbose=0)
module_transfer(x)

#from torchinfo import summary


# print(summary(original_model, (1, 3, 224, 224), device=torch.device('cpu')))
# 19,997,096
# print(summary(cvt_hugging_model, (1, 3, 224, 224), device=torch.device('cpu'))) 
# 19,997,096

original_logits = original_model(x.clone())
cvt_hugging_face_output = cvt_hugging_model(x.clone())


print('asddsa')