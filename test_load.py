import torch
import os

checkpoint_paths = {
    'backbone': '/tmp/backbone.pth',
    'linear_head': '/tmp/lc.pth'
}

for name, path in checkpoint_paths.items():
    if os.path.exists(path):
        try:
            checkpoint = torch.load(path, map_location='cpu')
            print(f"✓ Successfully loaded {name} checkpoint from {path}")
            print(f"  Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'tensor'}")
        except Exception as e:
            print(f"✗ Failed to load {name} checkpoint: {e}")
    else:
        print(f"✗ {name} checkpoint not found at {path}")


