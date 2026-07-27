#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from ccpfm_extended_diffusion_core import aggregate, solve_case, write_csv

Job = Tuple[str, str, str, str, float, int, int, str, bool]


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--base-script', default='ccpfm_poisson_validation_prior_preserving_core_reg_complete.py')
    p.add_argument('--outdir', default='extended_results')
    p.add_argument('--problems', nargs='+', default=['smooth', 'interface'], choices=['smooth', 'interface'])
    p.add_argument('--domains', nargs='+', default=['circle', 'ellipse', 'star', 'annulus'])
    p.add_argument('--h', nargs='+', type=float, default=[0.20, 0.16, 0.125, 0.10])
    p.add_argument('--trials', type=int, default=5)
    p.add_argument('--seed-base', type=int, default=8100)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--save-npz', action='store_true')
    p.add_argument('--resume', action=argparse.BooleanOptionalAction, default=True,
                   help='Reuse completed case JSON files and checkpoint CSV rows (default: true).')
    return p.parse_args()


def worker(args: Job) -> Dict[str, object]:
    return solve_case(*args)


def case_key(row: Dict[str, object]) -> Tuple[str, str, str, float, int, int]:
    return (
        str(row['problem']), str(row['domain']), str(row['method']),
        round(float(row['h']), 12), int(row['seed']), int(row['trial'])
    )


def job_key(job: Job) -> Tuple[str, str, str, float, int, int]:
    _, problem_key, domain, method, h, seed, trial, _, _ = job
    canonical = 'smooth_variable' if problem_key == 'smooth' else 'circular_interface'
    return canonical, domain, method, round(float(h), 12), seed, trial


def load_checkpoint(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    # Normalize fields needed by aggregate(). Other numeric fields remain strings
    # and are converted there with float().
    for row in rows:
        row['h'] = float(row['h'])
        row['seed'] = int(float(row['seed']))
        row['trial'] = int(float(row['trial']))
        row['unstable'] = str(row.get('unstable', '')).lower() in {'1', 'true', 'yes'}
    return rows


def checkpoint(rows: Sequence[Dict[str, object]], out: Path) -> None:
    write_csv(rows, out / 'case_results.csv')
    agg = aggregate(rows)
    write_csv(agg, out / 'aggregate_results.csv')
    (out / 'aggregate_results.json').write_text(json.dumps(agg, indent=2), encoding='utf-8')


def main() -> int:
    a = parse()
    root = Path(__file__).resolve().parent
    base = (root / a.base_script).resolve()
    out = (root / a.outdir).resolve()
    cases = out / 'cases'
    out.mkdir(parents=True, exist_ok=True)
    cases.mkdir(parents=True, exist_ok=True)

    jobs: List[Job] = []
    for problem_key in a.problems:
        domains = a.domains if problem_key == 'smooth' else ['circle']
        for domain in domains:
            for h in a.h:
                for trial in range(a.trials):
                    seed = a.seed_base + trial
                    for method in ('wls_strong', 'ccpfm_flux'):
                        jobs.append((str(base), problem_key, domain, method, h, seed,
                                     trial, str(cases), a.save_npz))

    rows = load_checkpoint(out / 'case_results.csv') if a.resume else []
    completed = {case_key(row) for row in rows}
    pending = [job for job in jobs if job_key(job) not in completed]

    print(f'Total cases: {len(jobs)}; completed: {len(jobs)-len(pending)}; pending: {len(pending)}')
    if not pending:
        checkpoint(rows, out)
        print('All requested cases are already complete.')
        return 0

    def accept(result: Dict[str, object]) -> None:
        rows.append(result)
        checkpoint(rows, out)
        done = len({case_key(row) for row in rows} & {job_key(job) for job in jobs})
        print(f'[{done}/{len(jobs)}] {result["problem"]} {result["domain"]} '
              f'{result["method"]} h={result["h"]} L2={result["l2"]:.3e}', flush=True)

    if a.workers == 1:
        for job in pending:
            accept(worker(job))
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futures = {ex.submit(worker, job): job for job in pending}
            try:
                for future in as_completed(futures):
                    accept(future.result())
            except BaseException:
                # Successful cases have already been checkpointed. Cancel work that
                # has not started and propagate the original error with its traceback.
                for future in futures:
                    future.cancel()
                raise

    checkpoint(rows, out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
