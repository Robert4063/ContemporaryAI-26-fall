"""Reproduce all 392 experiment configurations using the Python standard library."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
TESTS=['tests/test_search.py','tests/test_reusable_search.py','tests/test_direction_search.py']
EXPERIMENTS=['run_experiments.py','run_reuse_experiments.py','run_direction_experiments.py',
             'compare_direction_results.py']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--tests-only',action='store_true',help='Run the 35 unit tests only')
    mode.add_argument('--verify-only',action='store_true',help='Verify existing results without rerunning timed experiments')
    args=parser.parse_args()
    if sys.version_info<(3,10):parser.error('Python 3.10 or newer is required')
    (ROOT/'results').mkdir(exist_ok=True)
    steps=(['audit_experiments.py'] if args.verify_only else TESTS if args.tests_only else
           TESTS+EXPERIMENTS+['audit_experiments.py'])
    for index,script in enumerate(steps,1):
        print(f'[{index}/{len(steps)}] {sys.executable} {script}',flush=True)
        # Suppress bytecode artifacts in the source checkout on every platform.
        subprocess.run([sys.executable,'-B',str(ROOT/script)],cwd=ROOT,check=True)
    print('All requested steps passed.',flush=True)


if __name__=='__main__':
    try:main()
    except subprocess.CalledProcessError as error:sys.exit(error.returncode)
