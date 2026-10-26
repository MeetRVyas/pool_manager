"""
Run the simulated evaluation.

    python -m evals.run_eval                       # all scenarios, all implementations, 30 seeds
    python -m evals.run_eval --seeds 10 --scenarios chaos_70,steady_90
    python -m evals.run_eval --quick               # 5 seeds, core scenarios only (smoke test)
    python -m evals.run_eval --resume --impls X    # keep what is already in <out>/results.json, run only what is missing

Writes <out>/results.json (every per-seed metric) and <out>/REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DEFAULT_SEEDS = 30
WAIT_SCENARIOS = ("bursty_60", "steady_90", "chaos_70", "overload_150", "load_200")     # where waiting matters


def pending_cells(done: dict, wanted: list[tuple[str, str, list[int]]]) -> list[tuple[str, str, list[int]]]:
    """Cells not yet complete in `done` ({scenario: {impl: [per-seed metrics]}}). A cell is complete when it has one result per seed."""
    return [(n, i, s) for n, i, s in wanted if len(done.get(n, {}).get(i, [])) != len(s)]


def _run_cell(args: tuple[str, str, list[int]]) -> tuple[str, str, list[dict]]:
    scn_name, impl, seeds = args
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)            # the pool logs a warning per failover; we measure, not read
    from evals.scenarios import all_scenarios
    from evals.sim import simulate

    scn = all_scenarios()[scn_name]
    return scn_name, impl, [simulate(scn, impl, s) for s in seeds]


def main(argv: list[str] | None = None) -> int:
    from evals.impls import IMPLS, WAIT_IMPLS
    from evals.scenarios import all_scenarios, core_scenarios, sweep_scenarios

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--scenarios", default="all", help="'all', 'core', 'sweep' or a comma-separated list of names")
    ap.add_argument("--impls", default=",".join(IMPLS))
    ap.add_argument("--out", default="evals/results")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="keep the cells already in <out>/results.json (same seeds) and run only the missing ones; "
                         "results are deterministic per (scenario, implementation, seed), so this equals a one-shot run")
    ap.add_argument("--no-wait-sweep", action="store_true", help="skip the max_wait_seconds experiment")
    args = ap.parse_args(argv)

    names = {"all": [s.name for s in core_scenarios() + sweep_scenarios()],
             "core": [s.name for s in core_scenarios()],
             "sweep": [s.name for s in sweep_scenarios()]}.get(args.scenarios, args.scenarios.split(","))
    impls = args.impls.split(",")
    seeds = list(range(1, (5 if args.quick else args.seeds) + 1))
    if args.quick and args.scenarios == "all":
        names = [s.name for s in core_scenarios()]
    unknown = [n for n in names if n not in all_scenarios()] + [i for i in impls if i not in IMPLS]
    if unknown:
        ap.error(f"unknown scenario/impl: {unknown}")

    cells = [(n, i, seeds) for n in names for i in impls]
    wait_scns = [n for n in WAIT_SCENARIOS if n in names]
    wait_impls = [] if args.no_wait_sweep or "merged" not in impls else list(WAIT_IMPLS)
    cells += [(n, i, seeds) for n in wait_scns for i in wait_impls]
    out = Path(args.out)
    prior: dict = {}
    prior_wall = 0.0
    if args.resume and (out / "results.json").exists():
        saved = json.loads((out / "results.json").read_text())
        if saved["meta"]["seeds"] != seeds:
            ap.error(f"--resume: saved results use seeds 1..{len(saved['meta']['seeds'])}, this run uses 1..{len(seeds)}")
        prior, prior_wall = saved["runs"], float(saved["meta"].get("wall_seconds", 0.0))
    total_cells = len(cells)
    cells = pending_cells(prior, cells)
    print(f"{total_cells} cells wanted ({len(names)} scenarios x {len(impls)} implementations, plus {len(wait_scns) * len(wait_impls)} "
          f"wait-budget cells), {total_cells - len(cells)} already saved, {len(cells)} to run x {len(seeds)} seeds = "
          f"{len(cells) * len(seeds)} simulated runs on {args.workers} workers", flush=True)
    started = time.time()
    runs: dict[str, dict[str, list[dict]]] = {n: dict(v) for n, v in prior.items()}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for done, (scn, impl, res) in enumerate(ex.map(_run_cell, cells), 1):
            runs.setdefault(scn, {})[impl] = res
            if done % 10 == 0 or done == len(cells):
                print(f"  {done}/{len(cells)} cells  ({time.time() - started:.0f}s)", flush=True)

    import langchain_core, langchain_google_genai  # noqa: E401
    from evals.scenarios import MODELS, N_KEYS, TRUE_RPM
    every = all_scenarios()
    have_scn = [n for n in every if n in runs]
    have_impl = [i for i in IMPLS if any(i in runs[n] for n in runs)]
    have_wait = [i for i in WAIT_IMPLS if any(i in runs[n] for n in runs)]
    meta = {
        "seeds": seeds,
        "scenarios": {n: every[n].description for n in have_scn},
        "implementations": {i: IMPLS[i] for i in have_impl},
        "wait_scenarios": [n for n in WAIT_SCENARIOS if n in runs and any(w in runs[n] for w in have_wait)],
        "wait_implementations": {i: WAIT_IMPLS[i] for i in have_wait},
        "topology": {"models": list(MODELS), "keys": N_KEYS, "true_rpm_per_key": TRUE_RPM},
        "python": platform.python_version(),
        "langchain_core": langchain_core.__version__,
        "langchain_google_genai": langchain_google_genai.__version__ if hasattr(langchain_google_genai, "__version__") else "?",
        "wall_seconds": round(prior_wall + time.time() - started, 1),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"meta": meta, "runs": runs}, indent=1))

    from evals.report import render_report
    (out / "REPORT.md").write_text(render_report(meta, runs), encoding="utf-8")
    print(f"wrote {out / 'results.json'} and {out / 'REPORT.md'}  ({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
