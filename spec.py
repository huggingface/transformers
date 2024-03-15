
import torch
import intel_extension_for_pytorch
 
DEVICE_NAME = "xpu"
 
MANUAL_SEED_FN = torch.xpu.manual_seed
EMPTY_CACHE_FN = torch.xpu.empty_cache
DEVICE_COUNT_FN = torch.xpu.device_count
