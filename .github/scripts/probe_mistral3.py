"""Capture Mistral3 outputs under controlled execution modes, without recording baselines."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

OUT = Path('/tmp/mic-cause')
OUT.mkdir(exist_ok=True)
REVISION = 'fd12f1c1aeaa3e4a904ff1e37decc6381b499c65'

if len(sys.argv) == 1:
    # Each scenario starts in a fresh process. Full-fleet mode preserves the
    # production ordering and can reveal state left behind by earlier contracts.
    for mode in ('alone', 'isolate', 'fleet'):
        for repeat in range(2):
            subprocess.run([sys.executable, __file__, mode, str(repeat)], check=True)
    raise SystemExit(0)

import torch
from transformersci.contracts import run

mode, repeat = sys.argv[1:]
folder = OUT / f'{mode}-{repeat}'
folder.mkdir(exist_ok=True)
original = run.Executor.run
counter = itertools.count()

def state():
    return {
        'dtype': str(torch.get_default_dtype()),
        'threads': torch.get_num_threads(),
        'interop_threads': torch.get_num_interop_threads(),
        'matmul_precision': torch.get_float32_matmul_precision(),
        'deterministic': torch.are_deterministic_algorithms_enabled(),
        'grad_enabled': torch.is_grad_enabled(),
        'mkldnn': torch.backends.mkldnn.enabled,
        'hash_seed': os.environ.get('PYTHONHASHSEED'),
    }

def capture(self, env, fresh_process=False):
    before = state()
    result = original(self, env, fresh_process=fresh_process)
    (folder / f'call-{next(counter):03}.json').write_text(json.dumps({
        'entry': str(self.entry), 'env': env, 'fresh_process': fresh_process,
        'inline': self.inline, 'before': before, 'after': state(),
        'success': result[0], 'stdout': result[1], 'stderr': result[2],
    }, indent=2))
    return result

run.Executor.run = capture
args = ['--fixtures-only', '--fleet', 'prototypes', '--fleet-revision', REVISION,
        '--variant', 'bfloat16', '--output', str(folder / 'results.jsonl')]
if mode != 'fleet':
    args += ['--contract', 'mic-mistral3']
if mode == 'isolate':
    args += ['--isolate']
try:
    run.main(args)
except SystemExit as exc:
    print(mode, repeat, 'exit', exc.code, flush=True)
