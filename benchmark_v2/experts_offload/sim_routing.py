"""Routing locality on synthetic weights: random router weights pick experts uniformly; real routers favour some."""

import torch


def skew_routers(model, skew):
    """Routers' logits -> unit std + skew * log(1 / popularity rank), a random rank order per layer (seeded)."""
    generator = torch.Generator().manual_seed(0)
    count = 0
    for module in model.modules():
        if not type(module).__name__.endswith("TopKRouter"):
            continue
        ranks = torch.randperm(module.num_experts, generator=generator).float() + 1
        module.register_buffer("popularity", (-skew * ranks.log()).to(module.weight.device), persistent=False)

        def forward(hidden_states, module=module):
            hidden_states = hidden_states.reshape(-1, module.hidden_dim)
            logits = torch.nn.functional.linear(hidden_states, module.weight).float()
            logits = logits / logits.std(dim=-1, keepdim=True) + module.popularity
            probs = logits.softmax(dim=-1)
            top_value, indices = probs.topk(module.top_k, dim=-1)
            if module.norm_topk_prob:
                top_value = top_value / top_value.sum(dim=-1, keepdim=True)
            return logits, top_value.to(module.weight.dtype), indices

        module.forward = forward
        count += 1
    return count
