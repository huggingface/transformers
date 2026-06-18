# ... existing imports ...

# Phi-optimized generation
    PhiRecursiveGenerator,
    phi_max_tokens,
    phi_repetition_penalty,
    phi_temperature,
    phi_top_k,
    phi_top_p,
)

# ... rest of file ...

# Phi-optimized generation (lazy import to reduce complexity)
def PhiRecursiveGenerator(*args, **kwargs):
    return _PhiRecursiveGenerator(*args, **kwargs)
