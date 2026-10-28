"""Select a measured package source and sync with a recoverable lockfile swap."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import tomllib

INDEX = {'pypi': 'https://pypi.org/simple',
         'aliyun': 'https://mirrors.aliyun.com/pypi/simple/'}


def signature(data):
    """Ignore origin URLs, retain versions, markers, artifacts and hashes."""
    def normalize(value):
        if isinstance(value, dict):
            return {k: ('registry' if k == 'registry' else
                        v.split('/')[-1] if k == 'url' else normalize(v))
                    for k, v in value.items() if k not in ('size', 'upload-time')}
        if isinstance(value, list):
            return sorted((normalize(v) for v in value), key=repr)
        return value
    return normalize(copy.deepcopy(data))


def probe(url):
    # curl's total timeout also bounds DNS, TLS and slow streaming responses.
    result = subprocess.run([
        'curl', '--fail', '--silent', '--location', '--connect-timeout', '3',
        '--max-time', '6', '--range', '0-262143', '--max-filesize', '1048576',
        '--output', os.devnull, '--write-out', '%{speed_download}', url,
    ], capture_output=True, text=True, timeout=8)
    if result.returncode:
        return 0.0
    try:
        return max(0.0, float(result.stdout))
    except ValueError:
        return 0.0


def select(source, locks):
    if signature(locks['pypi']) != signature(locks['aliyun']):
        raise ValueError('Lockfiles differ in versions, dependencies or artifacts; regenerate both.')
    if source != 'auto':
        return source
    candidates = [w['url'].rsplit('/', 1)[-1]
                  for p in locks['pypi']['package'] for w in p.get('wheels', [])
                  if w['url'].endswith('-py3-none-any.whl') and 32768 <= w.get('size', 0) <= 262144]
    if not candidates:
        raise ValueError('No suitable wheel available for source measurement.')
    filename = sorted(candidates)[0]
    urls = {name: next(w['url'] for p in data['package'] for w in p.get('wheels', [])
                       if w['url'].rsplit('/', 1)[-1] == filename)
            for name, data in locks.items()}
    def measure(name):
        try:
            return probe(urls[name])
        except (OSError, subprocess.TimeoutExpired):
            return 0.0
    with ThreadPoolExecutor(max_workers=2) as pool:
        speeds = dict(zip(urls, pool.map(measure, urls)))
    for name, speed in speeds.items():
        print(f'{name}: {speed / 1024:.1f} KiB/s' if speed else f'{name}: probe failed', flush=True)
    if not any(speeds.values()):
        raise ValueError('Both source probes failed. Check network or use --source pypi/aliyun (cached/offline install).')
    return max(speeds, key=speeds.get)


def atomic_write(path, content):
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def refresh_mirror(python):
    """Resolve mirror metadata, then retain the canonical graph and artifact hashes."""
    import re
    import shutil
    original = Path('uv.lock').read_text()
    base = tomllib.loads(original)
    with tempfile.TemporaryDirectory(prefix='paimem-mirror-') as directory:
        for filename in ('pyproject.toml', 'uv.lock'):
            shutil.copyfile(filename, Path(directory) / filename)
        subprocess.run(['uv', 'lock', '--directory', directory, '--default-index', INDEX['aliyun'],
                        '--python', python, '--no-python-downloads'], check=True)
        mirror = tomllib.loads((Path(directory) / 'uv.lock').read_text())
    packages = {(p['name'], p['version']): p for p in mirror['package']}
    if set(packages) != {(p['name'], p['version']) for p in base['package']}:
        raise ValueError('Mirror resolution changed package versions; mirror lock was not written.')
    urls = {}
    def artifacts(package):
        return package.get('wheels', []) + ([package['sdist']] if 'sdist' in package else [])
    for package in base['package']:
        mirrored = packages[package['name'], package['version']]
        by_hash = {w['hash']: w['url'] for w in artifacts(mirrored)}
        for wheel in artifacts(package):
            if wheel['hash'] not in by_hash:
                raise ValueError(f"Mirror missing matching artifact for {package['name']}; not written.")
            urls[wheel['url']] = by_hash[wheel['hash']]
    result = re.sub(r'url = "([^"]+)"', lambda m: 'url = "' + urls[m[1]] + '"', original)
    result = result.replace('registry = "https://pypi.org/simple"',
                            'registry = "' + INDEX['aliyun'] + '"')
    if signature(base) != signature(tomllib.loads(result)):
        raise ValueError('Generated mirror lock differs from canonical lock.')
    atomic_write(Path('uv.aliyun.lock'), result.encode())
    print('Updated uv.aliyun.lock; canonical versions, graph and hashes preserved.')


def sync(source, python):
    root = Path.cwd()
    # Keep the guard outside Git; never unlink a flock file while waiters may exist.
    import hashlib
    guard_path = Path('/tmp') / ('paimem-setup-' + hashlib.sha256(str(root).encode()).hexdigest() + '.lock')
    with guard_path.open('a') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another service setup is running in this checkout.')
        current = root / 'uv.lock'
        backup = root / '.uv.lock.setup-backup'
        if backup.exists():
            print('Recovering original uv.lock from interrupted setup.', flush=True)
            os.replace(backup, current)
        original = current.read_bytes()
        mirror = (root / 'uv.aliyun.lock').read_bytes()
        chosen = select(source, {'pypi': tomllib.loads(original.decode()),
                                 'aliyun': tomllib.loads(mirror.decode())})
        print(f'Service package source: {chosen}', flush=True)
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        try:
            if chosen == 'aliyun':
                atomic_write(backup, original)
                atomic_write(current, mirror)
            return subprocess.call(['uv', 'sync', '--python', python, '--frozen',
                                    '--default-index', INDEX[chosen]],
                                   env={**os.environ, 'UV_PROJECT_ENVIRONMENT': str(root / '.venv')},
                                   pass_fds=(guard.fileno(),))
        finally:
            if backup.exists():
                os.replace(backup, current)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['auto', 'pypi', 'aliyun'], default='auto')
    parser.add_argument('--python', default='3.11')
    parser.add_argument('--refresh-mirror', action='store_true')
    args = parser.parse_args()
    try:
        if args.refresh_mirror:
            refresh_mirror(args.python)
        else:
            sys.exit(sync(args.source, args.python))
    except (ValueError, OSError) as exc:
        sys.exit(str(exc))
    except KeyboardInterrupt:
        sys.exit(130)
