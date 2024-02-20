import logging
import inspect
import torch
from torch.export import export
import torch._dynamo as torchdynamo
from functorch.experimental.control_flow import cond

torchdynamo.config.capture_scalar_outputs = True



class MyModule():
    def __init__(self):
        pass

    def __call__(self, y):
        return y * 2


class CondBranchClassMethod():
    def __init__(self):
        self.subm = MyModule()

    def __call__(self, x):
        if hasattr(self, "get_compiler_config"):
            logging.warning("the function is compiled")
        if x.shape[-1] < 10:
            return x
        return x + 1


class CondOperands(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        def true_fn(x, y):
            return x + y

        def false_fn(x, y):
            return x * y

        a = cond(x.shape[0] > 2, true_fn, false_fn, [x, y])
        a = a.to(dtype=torch.float)
        return a


module = CondOperands()
real_result = module(torch.tensor([1]), torch.tensor([10]))

example_args = (torch.randint(0, 10, size=(1,)), torch.randint(0, 10, size=(1,)))
exported_program = export(module, example_args)
export_res = exported_program(torch.tensor([1]), torch.tensor([10]))
print(export_res)


cond_mod = CondBranchClassMethod()
def dummy_func(inputs):
    output = cond_mod(inputs)
    output = output / 2
    output = torch.where(output == 0, -1, output)
    return output

cond_module_compiled = torch.compile(dummy_func, mode="reduce-overhead", fullgraph=True)
print(cond_module_compiled)
print( [i for i in cond_module_compiled.__dict__.keys() if i[:1] != '_'])
print( [i for i in cond_mod.__dict__.keys() if i[:1] != '_'])

x = cond_module_compiled(torch.tensor([1], device="cuda:0"))


#module_compiled = torch.compile(module, mode="reduce-overhead", fullgraph=True)
#com_result = module_compiled(torch.tensor([1]), torch.tensor([10]))

#a = torch.tensor([1], dtype=torch.float)
#b = torch.tensor([10], dtype=torch.float)
#for i in range(10):
#    com_result = module_compiled(a, b)
#    print(com_result)
#    a += 1
