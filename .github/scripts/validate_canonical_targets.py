"""Compare frozen fleets around a target-only baseline rename on native CPU."""
import collections
import json
import os
from pathlib import Path
import subprocess
import sys

from huggingface_hub import hf_hub_download
import torch
import transformers

out = Path('/tmp/mic-target-validation')
assert Path(transformers.__file__).resolve() == Path('src/transformers/__init__.py').resolve()
assert torch.get_num_threads() == 4
revisions = {'before': os.environ['BEFORE_REVISION'], 'after': os.environ['CANDIDATE_REVISION']}
renames = {}
all_rows = {}
for phase, revision in revisions.items():
    output = out / f'{phase}.jsonl'
    output.unlink(missing_ok=True)
    for fleet, filename, variant in [('control', 'fleet-lock.json', 'default'), ('prototypes', 'prototype-fleet.lock', 'bfloat16')]:
        lock = json.loads(Path(hf_hub_download('huggingface-mic/transformers-fleet', filename, repo_type='dataset', revision=revision)).read_text())
        for name, contract in lock['contracts'].items():
            for repo, pin in contract['fixtures'].items():
                renames[phase, name, pin.get('target', repo.split('/')[-1])] = repo.split('/')[-1]
        result = subprocess.run([sys.executable, '-m', 'transformersci.contracts.run', '--fixtures-only', '--fleet', fleet, '--fleet-revision', revision, '--variant', variant, '--output', str(output)])
        print(phase, fleet, 'exit', result.returncode, flush=True)
    rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    assert len(rows) == 31, (phase, len(rows))
    assert all(row['fleet_revision'] == revision for row in rows)
    all_rows[phase] = {(row['contract_name'], renames[phase, row['contract_name'], row['target']], row['variant']): row for row in rows}
assert all_rows['before'].keys() == all_rows['after'].keys()
changes = []
for key, after in all_rows['after'].items():
    before = all_rows['before'][key]
    assert not any('no reviewed baseline' in text for text in after.get('failures', [])), after
    if before['status'] != after['status']:
        changes.append({'case': key, 'before': before, 'after': after})
summary = {'revisions': revisions, 'cases': len(all_rows['after']), 'statuses': {phase: dict(collections.Counter(row['status'] for row in rows.values())) for phase, rows in all_rows.items()}, 'changed_outcomes': changes, 'renamed': [{'contract': key[0], 'old_target': all_rows['before'][key]['target'], 'target': row['target']} for key, row in all_rows['after'].items() if all_rows['before'][key]['target'] != row['target']]}
(out / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary), flush=True)
assert not changes, 'Rename changed case outcomes; inspect the before/after records'
assert len(summary['renamed']) == 20, len(summary['renamed'])
