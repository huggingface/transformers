from fairseq.quantization.utils.quant_modules import *

def freeze_model(model):
    """
    freeze the activation range. Resursively invokes layer.fix()
    """
    if type(model) in [QuantAct, QuantLinear, IntLayerNorm, IntSoftmax]:
        model.fix()
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            freeze_model(m)
    elif type(model) == nn.ModuleList:
        for n in model:
            freeze_model(n)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                freeze_model(mod)

def unfreeze_model(model):
    """
    unfreeze the activation range. Resursively invokes layer.unfix()
    """
    if type(model) in [QuantAct, QuantLinear, IntLayerNorm, IntSoftmax]:
        model.unfix()
    elif type(model) == nn.Sequential:
        for n, m in model.named_children():
            unfreeze_model(m)
    elif type(model) == nn.ModuleList:
        for n in model:
            unfreeze_model(n)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module):
                unfreeze_model(mod)
