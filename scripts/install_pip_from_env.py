import yaml
import subprocess
import sys
p='environment.yml'
with open(p) as f:
    data = yaml.safe_load(f)
# Extract pip list
pip_list = []
for dep in data.get('dependencies', []):
    if isinstance(dep, dict) and 'pip' in dep:
        pip_list = dep['pip']
        break
skip_patterns = ['nvidia-', 'triton', 'torch==', 'torch-geometric', 'torchvision', 'mujoco', 'mujoco-py', 'dm-', 'av==', 'imagecodecs', 'llvmlite==0.42.0']
log = []
for pkg in pip_list:
    if any(pat in pkg for pat in skip_patterns):
        log.append((pkg,'SKIPPED','pattern'))
        continue
    print('\nINSTALLING:', pkg)
    proc = subprocess.run([sys.executable, '-m', 'pip', 'install', pkg], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode == 0:
        log.append((pkg,'OK',''))
        print('OK')
    else:
        err = proc.stderr.strip() or proc.stdout.strip()
        short = err.splitlines()[-1] if err else ''
        log.append((pkg,'FAILED', short))
        print('FAILED:', short)
# Summary
ok = [x for x in log if x[1]=='OK']
failed = [x for x in log if x[1]=='FAILED']
skipped = [x for x in log if x[1]=='SKIPPED']
print('\nSUMMARY:')
print('  OK:', len(ok))
print('  FAILED:', len(failed))
print('  SKIPPED:', len(skipped))
if failed:
    print('\nFailures:')
    for f in failed[:40]:
        print(' -', f[0], f[2])
