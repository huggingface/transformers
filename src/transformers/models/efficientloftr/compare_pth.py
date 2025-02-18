import torch


modu_in = torch.load("modu_hidden_states.pth").reshape(2, 2, -1, 256)[:, 0]
orig_in = torch.load("orig_hidden_states")

modu_param = torch.load("modu_q_proj.pth")
orig_param = torch.load("orig_q_proj.pth")


modu = modu_in @ modu_param
orig = orig_in @ orig_param

print(torch.allclose(a, b))

