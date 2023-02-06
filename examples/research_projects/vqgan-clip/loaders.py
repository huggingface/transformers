import importlib

import torch
import yaml
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_vqgan(device, conf_path=None, ckpt_path=None):
    if conf_path is None:
        conf_path = "./model_checkpoints/vqgan_only.yaml"
    config = load_config(conf_path, display=False)
    model = VQModel(**config.model.params)
    if ckpt_path is None:
        ckpt_path = "./model_checkpoints/vqgan_only.pt"
    sd = torch.load(ckpt_path, map_location=device)
    if ".ckpt" in ckpt_path:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.to(device)
    del sd
    return model


def reconstruct_with_vqgan(x, model):
    z, _, [_, _, indices] = model.encode(x)
    print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
    xrec = model.decode(z)
    return xrec


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    model = instantiate_from_config(config)
    if sd is not None:
        model.load_state_dict(sd)
    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return {"model": model}


def load_model(config, ckpt, gpu, eval_mode):
    # load the specified checkpoint
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
        print(f"loaded model from global step {global_step}.")
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model, pl_sd["state_dict"], gpu=gpu, eval_mode=eval_mode)["model"]
    return model, global_step
