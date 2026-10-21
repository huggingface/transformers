"""Prove calendar-dependent inputs, then test an explicit stable system prompt."""
from datetime import datetime as RealDatetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OUT = Path('/tmp/mic-date-proof')
OUT.mkdir(exist_ok=True)
if len(sys.argv) == 1:
    for prompt in ('original', 'explicit'):
        for date in ('2026-10-20', '2026-10-21', '2026-10-22'):
            subprocess.run([sys.executable, __file__, prompt, date], check=True)
    results = [json.loads(p.read_text()) for p in sorted(OUT.glob('*.json'))]
    explicit = [r for r in results if r['prompt'] == 'explicit']
    assert all(r['outputs'] == explicit[0]['outputs'] for r in explicit)
    assert all(r['inputs'] == explicit[0]['inputs'] for r in explicit)
    print('Explicit prompt inputs and outputs match exactly across all three dates', flush=True)
    from transformersci.contracts import run
    result_path = OUT / 'fleet-results.jsonl'
    for fleet, variant in [('control', 'default'), ('prototypes', 'bfloat16')]:
        try:
            run.main(['--fixtures-only', '--fleet', fleet, '--fleet-revision', os.environ['CANDIDATE_REVISION'], '--variant', variant, '--output', str(result_path)])
        except SystemExit as exc:
            assert exc.code in (None, 0), (fleet, exc.code)
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    assert len(rows) == 31
    assert all(row['status'] == 'pass' for row in rows), rows
    print('Candidate fleet: 31/31 pass', flush=True)
    raise SystemExit(0)

from huggingface_hub import snapshot_download
from transformers.utils import chat_template_utils
from transformersci.contracts.run import contract_code, execute_inline

prompt, date = sys.argv[1:]
class FrozenDatetime(RealDatetime):
    @classmethod
    def now(cls, tz=None):
        return cls.fromisoformat(date + 'T12:00:00')
chat_template_utils.datetime = FrozenDatetime
folder = Path(snapshot_download('huggingface-mic/mistral3-mic', revision='6320652c65d41c49903fe19d7cc2e5c0ab2842ad'))
code = contract_code(folder / 'integrations/transformers/usage.md')
if prompt == 'explicit':
    code = code.replace('processor = AutoProcessor.from_pretrained(model_id)', 'processor = AutoProcessor.from_pretrained(model_id)\nprocessor.chat_template = processor.chat_template.replace(\'strftime_now("%Y-%m-%d")\', \'"2026-10-20"\')')
code = code.replace('model.eval()', 'print("INPUTS=" + json.dumps({k: v.tolist() for k, v in inputs.items()}))\nprint("TEXTS=" + json.dumps(texts))\nmodel.eval()')
ok, stdout, stderr = execute_inline(code, str(folder / 'integrations/transformers/usage.md'), {
    'MIC_RUNNER': '1', 'MIC_MODEL': str(folder), 'MIC_MODEL_KWARGS': '{"dtype": "bfloat16"}',
})
assert ok, stderr
lines = stdout.splitlines()
outputs = next(json.loads(line) for line in reversed(lines) if line.startswith('{'))
inputs = next(json.loads(line.removeprefix('INPUTS=')) for line in lines if line.startswith('INPUTS='))
texts = next(json.loads(line.removeprefix('TEXTS=')) for line in lines if line.startswith('TEXTS='))
baseline = json.loads((folder / 'integrations/transformers/expectations/cpu-x86_64-bf16/mistral3-mic.bfloat16.json').read_text())['outputs']
if prompt == 'explicit' or date == '2026-10-20':
    assert outputs == baseline, (outputs, baseline)
row = {'prompt': prompt, 'date': date, 'outputs': outputs, 'inputs': inputs, 'texts': texts,
       'matches_reviewed_baseline_exactly': outputs == baseline}
(OUT / f'{prompt}-{date}.json').write_text(json.dumps(row, indent=2) + '\n')
print(prompt, date, outputs['answer_logits'][0][:4], outputs['training']['loss'], 'baseline exact:', outputs == baseline, flush=True)
