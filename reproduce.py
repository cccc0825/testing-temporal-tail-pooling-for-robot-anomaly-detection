"""Verify packaged evidence or regenerate diagnostics from frozen scores."""
from pathlib import Path
import argparse, hashlib, json, subprocess, sys
ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['verify', 'analysis', 'figures', 'tables'])
    args = parser.parse_args()
    if args.mode == 'verify':
        manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
        bad = [name for name, digest in manifest['files'].items()
               if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
        if bad:
            raise SystemExit('File checksum mismatch: ' + ', '.join(bad))
        print(f"Verified {len(manifest['files'])} release files.")
    elif args.mode == 'analysis':
        subprocess.run([sys.executable, str(ROOT / 'analysis/reanalyze.py'), '--compute-only'], check=True)
        for reference in sorted((ROOT / 'results/reference').glob('*.csv')):
            output = ROOT / 'analysis' / reference.name
            if sha(output) != sha(reference):
                raise SystemExit('Numerical output differs from the archive: ' + reference.name)
        print('All six numerical CSVs reproduce byte for byte from packaged scores.')
    elif args.mode == 'tables':
        subprocess.run([sys.executable, str(ROOT / 'analysis/export_tables.py')], check=True)
    else:
        subprocess.run([sys.executable, str(ROOT / 'analysis/plot_vector.py')], check=True)

if __name__ == '__main__':
    main()
