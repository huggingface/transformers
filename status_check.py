#!/usr/bin/env python3
import os
import torch
from transformers import AutoProcessor

print('🔧 Setup Status:')
print(f'  Transformers: ✅')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {"✅" if torch.cuda.is_available() else "❌"}')
print(f'  RAG fix: {"✅" if os.path.exists("examples/rag/README.md") else "❌"}')
print('🚀 Ready to proceed!')