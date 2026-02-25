import subprocess, re, sys

result = subprocess.run(
    [r'.\venv\Scripts\python.exe', '-X', 'importtime', '-c', 'import transformers'],
    capture_output=True, text=True, cwd=r'd:\Vishal\Coding\Contribute\transformers'
)
lines = result.stderr.splitlines()
entries = []
for line in lines:
    m = re.match(r'import time:\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(.*)', line)
    if m:
        self_us = int(m.group(1))
        cum_us = int(m.group(2))
        mod = m.group(3).strip()
        entries.append((cum_us, self_us, mod))

entries.sort(reverse=True)
print(f'Total entries: {len(entries)}')
print()
print('Top 40 by cumulative time:')
for cum, self_t, mod in entries[:40]:
    print(f'  cum={cum:8d}us  self={self_t:8d}us  {mod}')

print()
print('Top 40 by self time:')
entries_self = sorted(entries, key=lambda x: x[1], reverse=True)
for cum, self_t, mod in entries_self[:40]:
    print(f'  self={self_t:8d}us  cum={cum:8d}us  {mod}')

# Look for transformers-related modules
print()
print('Transformers-related modules:')
for cum, self_t, mod in entries:
    if 'transformers' in mod.lower():
        print(f'  cum={cum:8d}us  self={self_t:8d}us  {mod}')
