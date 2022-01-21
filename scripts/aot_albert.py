from transformers import AutoConfig, AutoModelForMaskedLM
import transformers
import torch
from functorch.compile import memory_efficient_fusion, aot_module, draw_graph_compile, nop
import torch.utils._pytree as pytree
import time
from torch import optim
import torch.nn as nn
from torch.nn.utils import _stateless
import pandas as pd

device = 'cuda'
torch.manual_seed(42)
config = AutoConfig.from_pretrained("albert-base-v2")
config.classifier_dropout_prob = 1e-15 # To have the computation but not the stochasticity
model = AutoModelForMaskedLM.from_config(config).to(device)

input_ids = torch.randint(0, config.vocab_size, (8, 512)).to(device)
decoder_ids = torch.randint(0, config.vocab_size, (8, 512)).to(device)


train_inputs = {'input_ids': input_ids, 'labels': decoder_ids}


pytree._register_pytree_node(transformers.modeling_outputs.MaskedLMOutput, lambda x: ([x.loss, x.logits], None), lambda values, _: transformers.modeling_outputs.MaskedLMOutput(loss=values[0], logits=values[1])) # Used for

def get_cur_memory():
    torch.cuda.synchronize()
    import gc; gc.collect()
    torch.cuda.empty_cache()
    stats = torch.cuda.memory_stats()
    peak_bytes_requirement = stats["allocated_bytes.all.current"]
    print(f"Current memory requirement: {peak_bytes_requirement / 1024 ** 3:.2f} GB")
    return peak_bytes_requirement

def bench_model(name, mod):
    m = None
    for i in range(5):
        out = mod(**train_inputs).loss.abs().sum()
        if i == 4:
            m = get_cur_memory()
        out.backward()
    iters = 20
    torch.cuda.synchronize()
    begin = time.time()
    for _ in range(iters):
        mod(**train_inputs).loss.sum().backward()
    torch.cuda.synchronize()
    t = (time.time()-begin)/iters
    print(name, (time.time()-begin)/iters)
    return t, m

results = []
aot_model = memory_efficient_fusion(model)
t,m = bench_model("eager", model)
nh, th, mh, tp, mp = "name", "time (s)", "mem (GB)", "time %", "mem %"
results.append({nh:"eager", th:t, mh:m/2**30})
with torch.jit.fuser("fuser2"):
    t,m = bench_model("aot", aot_model)
results.append({nh:"aot", th:t, mh:m/2**30})

# calculate relative improvements
base_r = results[0]
for r in results:
    r[mp] = round(100 * (r[th] - base_r[th]) / base_r[th])
    r[tp] = round(100 * (r[mh] - base_r[mh]) / base_r[mh])

print(pd.DataFrame(results).to_markdown(index=False, floatfmt=".3f"))

print()
## Checking correctness
def clear_params(model):
    for i in model.parameters(): i.grad = None

clear_params(model)
torch.manual_seed(0)
out1 = model(**train_inputs).loss
out1.sum().backward()
grad1 = [i.grad for i in model.parameters()]

clear_params(aot_model)
torch.manual_seed(0)
out2 = aot_model(**train_inputs).loss
out2.sum().backward()
grad2 = [i.grad for i in aot_model.parameters()]
print((out1 - out2).abs().max())
print(max([(a-b).abs().max() for a, b in zip(grad1, grad2)]))
