"""Run MCA201P Labs 06, 07, 09 and 10 (and optionally 08 MNIST)."""
from pathlib import Path
import argparse
import runpy
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLBACKEND', 'Agg')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--include-mnist',action='store_true',
                   help='Also run Lab 08 (requires TensorFlow; downloads MNIST first time).')
    args = p.parse_args()
    for lab in ([6,7,8,9,10] if args.include_mnist else [6,7,9,10]):
        print(f'\n--- Executing Lab {lab:02d} ---',flush=True)
        runpy.run_path(str(ROOT/f'lab_{lab:02d}.py'),run_name='__main__')
    print('\nSelected new practicals completed successfully.')


if __name__=='__main__':
    main()
