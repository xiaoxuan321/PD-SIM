# -*- coding: utf-8 -*-
"""Evaluate existing DPSIM simulation CSVs exactly through DeepSimulator's
original evaluation path for day_hour_emd, mae and dl.

Goal
----
Given the same ``log_test.xes`` and the same exported simulation CSV that came
from ``DeepSimulator._export_log()``, reproduce the inputs that were passed to
``DeepSimulator._evaluate_logs()`` and then call that original method directly.

Important note about exact reproducibility
------------------------------------------
The project's SimilarityEvaluator randomly samples test traces after trimming
20% from each side of the simulation traces.  The original DeepSimulator does
NOT reseed immediately before evaluation; its Python ``random`` state has
already been consumed by simulation generation/resource assignment. Therefore
an old historical result cannot, in general, be reproduced bit-for-bit from
only log_test.xes + simulation CSV unless the same sampled test case IDs (or the
same Python random state) are also known.

This script supports an optional --sample-manifest CSV to force the exact test
case subset used by an original run.  The manifest format is:

    dataset,run_num,caseid
    Production,1,Case123
    Production,1,Case456
    ...

The script always writes evaluation_sample_manifest.csv containing the case IDs
sampled in the current run, so a run can be reproduced later.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import inspect
import itertools as it
import multiprocessing as mp
import random
import re
import sys
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_METRICS = ("day_hour_emd", "mae", "dl")
DEFAULT_DPSIM_ROOT = Path("input_files") / "dpsim"
DEFAULT_TEST_ROOT = Path("input_files") / "event_logs" / "exp_data"
DEFAULT_OUTPUT_DIR = DEFAULT_DPSIM_ROOT / "evaluation_results_exact"
RUNS = tuple(range(1, 6))


def import_original_deep_simulator():
    """Import the exact evaluation entry point used by the original project."""
    import deep_simulator as ds
    from external_tools.SimilarityEvaluator import SimilarityEvaluator

    evaluator_path = Path(inspect.getfile(SimilarityEvaluator)).resolve()
    deep_path = Path(inspect.getfile(ds)).resolve()
    print(f"[code] deep_simulator: {deep_path}")
    print(f"[code] SimilarityEvaluator: {evaluator_path}")

    # Do not silently fall back to a different local SimilarityEvaluator.py.
    normalized = str(evaluator_path).replace("\\", "/")
    if "/external_tools/" not in normalized:
        raise RuntimeError(
            "The imported SimilarityEvaluator is not external_tools/SimilarityEvaluator.py: "
            f"{evaluator_path}"
        )
    return ds


def _strip_timezone_without_conversion(series: pd.Series) -> pd.Series:
    """Mirror deep_simulator._read_exp_xes: tz_localize(None), not UTC convert."""
    parsed = pd.to_datetime(series, errors="raise")
    try:
        if hasattr(parsed.dtype, "tz") and parsed.dt.tz is not None:
            parsed = parsed.dt.tz_localize(None)
    except (AttributeError, TypeError):
        # For unusual object dtype timestamp columns, normalize element-wise
        # while preserving wall-clock time rather than converting to UTC.
        def strip_one(value):
            ts = pd.Timestamp(value)
            return ts.tz_localize(None) if ts.tzinfo is not None else ts

        parsed = parsed.map(strip_one)
    return parsed


def read_test_xes_like_deep_simulator(path: Path, *, one_timestamp: bool = False) -> pd.DataFrame:
    """Reproduce the DataFrame stored in ``DeepSimulator.log_test``.

    This mirrors the uploaded project's ``_read_exp_xes`` plus the global sort
    performed in ``_read_inputs``.  Virtual Start/End rows are intentionally
    added because that is what ``self.log_test`` contains before
    ``_evaluate_logs`` removes them.
    """
    try:
        from pm4py import read_xes as pm4py_read_xes
    except ImportError as exc:
        raise ImportError(
            "Reading log_test.xes requires pm4py from the same project environment."
        ) from exc

    print(f"[read] test XES exactly like DeepSimulator: {path}")
    df = pm4py_read_xes(str(path))

    rename_map = {
        "case:concept:name": "caseid",
        "concept:name": "task",
        "org:resource": "user",
    }
    if "start:timestamp" in df.columns:
        rename_map["start:timestamp"] = "start_timestamp"
    if "time:timestamp" in df.columns:
        rename_map["time:timestamp"] = "end_timestamp"
    df = df.rename(columns=rename_map)

    required = ["caseid", "task", "end_timestamp"]
    if not one_timestamp:
        required.append("start_timestamp")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Test XES is missing required columns {missing}; found {list(df.columns)}")

    if "user" not in df.columns:
        df["user"] = ""

    # Exact _read_exp_xes boundary filtering (case-sensitive list).
    df = df[~df["task"].isin(["Start", "End", "start", "end"])].reset_index(drop=True)

    ts_cols = ["end_timestamp"] if one_timestamp else ["start_timestamp", "end_timestamp"]
    for col in ts_cols:
        df[col] = _strip_timezone_without_conversion(df[col])

    # pipeline/read_properties uses filter_d_attrib=True for this path; mirror
    # the exact necessary-column reduction from _read_exp_xes.
    keep_cols = ["caseid", "task", "user"] + ts_cols
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Add the same virtual Start/End records created by _read_exp_xes.
    records = df.to_dict("records")
    end_start_times: dict[tuple[object, str], pd.Timestamp] = {}
    for case, group in df.groupby("caseid"):
        if one_timestamp:
            end_start_times[(case, "Start")] = group["end_timestamp"].min() - timedelta(microseconds=1)
        else:
            end_start_times[(case, "Start")] = group["start_timestamp"].min() - timedelta(microseconds=1)
        end_start_times[(case, "End")] = group["end_timestamp"].max() + timedelta(microseconds=1)

    new_data: list[dict] = []
    records_sorted = sorted(records, key=lambda x: x["caseid"])
    for ckey, group in it.groupby(records_sorted, key=lambda x: x["caseid"]):
        trace = list(group)
        if not trace:
            continue
        start_event = {
            "caseid": trace[0]["caseid"],
            "task": "Start",
            "user": "Start",
            "end_timestamp": end_start_times[(ckey, "Start")],
        }
        if not one_timestamp:
            start_event["start_timestamp"] = end_start_times[(ckey, "Start")]
        trace.insert(0, start_event)

        end_event = {
            "caseid": trace[-1]["caseid"],
            "task": "End",
            "user": "End",
            "end_timestamp": end_start_times[(ckey, "End")],
        }
        if not one_timestamp:
            end_event["start_timestamp"] = end_start_times[(ckey, "End")]
        trace.append(end_event)
        new_data.extend(trace)

    key = "end_timestamp" if one_timestamp else "start_timestamp"
    test_sorted = pd.DataFrame(new_data).sort_values(key, ascending=True).reset_index(drop=True)
    return test_sorted


def read_test_csv_like_deep_simulator(path: Path, *, one_timestamp: bool = False) -> pd.DataFrame:
    """Best-effort CSV equivalent for a test log already exported in project schema."""
    print(f"[read] test CSV: {path}")
    df = pd.read_csv(path)
    rename_map = {
        "case:concept:name": "caseid",
        "concept:name": "task",
        "org:resource": "user",
        "resource": "user",
        "start:timestamp": "start_timestamp",
        "time:timestamp": "end_timestamp",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns})
    if "user" not in df.columns:
        df["user"] = ""
    df = df[~df["task"].isin(["Start", "End", "start", "end"])].reset_index(drop=True)
    ts_cols = ["end_timestamp"] if one_timestamp else ["start_timestamp", "end_timestamp"]
    for col in ts_cols:
        df[col] = _strip_timezone_without_conversion(df[col])
    keep_cols = ["caseid", "task", "user"] + ts_cols
    df = df[keep_cols].copy()

    # Reuse the exact virtual-boundary construction by constructing from rows.
    records = df.to_dict("records")
    end_start_times = {}
    for case, group in df.groupby("caseid"):
        anchor = group["end_timestamp"].min() if one_timestamp else group["start_timestamp"].min()
        end_start_times[(case, "Start")] = anchor - timedelta(microseconds=1)
        end_start_times[(case, "End")] = group["end_timestamp"].max() + timedelta(microseconds=1)
    new_data = []
    for ckey, group in it.groupby(sorted(records, key=lambda x: x["caseid"]), key=lambda x: x["caseid"]):
        trace = list(group)
        start_event = {"caseid": trace[0]["caseid"], "task": "Start", "user": "Start",
                       "end_timestamp": end_start_times[(ckey, "Start")]}
        end_event = {"caseid": trace[-1]["caseid"], "task": "End", "user": "End",
                     "end_timestamp": end_start_times[(ckey, "End")]}
        if not one_timestamp:
            start_event["start_timestamp"] = end_start_times[(ckey, "Start")]
            end_event["start_timestamp"] = end_start_times[(ckey, "End")]
        new_data.extend([start_event, *trace, end_event])
    key = "end_timestamp" if one_timestamp else "start_timestamp"
    return pd.DataFrame(new_data).sort_values(key, ascending=True).reset_index(drop=True)


def read_test_log(path: Path, *, one_timestamp: bool = False) -> pd.DataFrame:
    if path.suffix.lower() == ".xes":
        return read_test_xes_like_deep_simulator(path, one_timestamp=one_timestamp)
    if path.suffix.lower() == ".csv":
        return read_test_csv_like_deep_simulator(path, one_timestamp=one_timestamp)
    raise ValueError(f"Unsupported test log format: {path}")


def read_simulation_csv_like_exported_event_log(path: Path, *, one_timestamp: bool = False) -> pd.DataFrame:
    """Read an exported simulation CSV without the old standalone normalization.

    Original DeepSimulator evaluates the in-memory event_log directly.  Its CSV
    export only serializes that DataFrame.  To reconstruct it, preserve row
    order, case IDs, tasks, resources and all non-time columns; only parse the
    timestamp columns back to pandas datetimes.
    """
    print(f"[read] simulation CSV as exported event_log: {path}")
    df = pd.read_csv(path)

    required = ["caseid", "task", "end_timestamp"]
    if not one_timestamp:
        required.append("start_timestamp")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Simulation CSV must be the CSV exported from the original event_log. "
            f"Missing {missing}; found {list(df.columns)}"
        )

    ts_cols = ["end_timestamp"] if one_timestamp else ["start_timestamp", "end_timestamp"]
    for col in ts_cols:
        df[col] = _strip_timezone_without_conversion(df[col])

    # Crucially: no Start/End removal, no caseid prefixing, no sorting, no
    # deletion of processing/waiting columns, and no forced resource column.
    return df


def find_test_log(dataset: str, test_root: Path) -> Path:
    candidates = [
        test_root / dataset / "log_test.xes",
        test_root / dataset / "log_test.csv",
        Path("input_files") / "event_logs" / dataset / "log_test.xes",
        Path("input_files") / "event_logs" / dataset / "log_test.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No test log found for dataset '{dataset}'. Tried:\n  " + "\n  ".join(str(p) for p in candidates)
    )


def _extract_run_number(path: Path) -> int | None:
    """Extract DPSIM run number from gen_<dataset>_<run>.csv."""
    match = re.search(r"_(\d+)\.csv$", path.name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)\.csv$", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def find_simulation_files(dataset_dir: Path) -> dict[int, Path]:
    """Find the five original DPSIM exports gen_<dataset>_1.csv ... _5.csv."""
    dataset = dataset_dir.name
    exact = {run: dataset_dir / f"gen_{dataset}_{run}.csv" for run in RUNS}
    if all(path.is_file() for path in exact.values()):
        return exact

    found: dict[int, Path] = {}
    for path in sorted(dataset_dir.glob("gen_*.csv")):
        run = _extract_run_number(path)
        if run in RUNS:
            if run in found:
                raise ValueError(f"Ambiguous DPSIM run {run}: {found[run]} and {path}")
            found[run] = path
    return found


def discover_datasets(dpsim_root: Path, selected: list[str] | None):
    if not dpsim_root.exists():
        raise FileNotFoundError(f"DPSIM root does not exist: {dpsim_root}")
    dataset_dirs = [dpsim_root / name for name in selected] if selected else sorted(
        p for p in dpsim_root.iterdir() if p.is_dir() and not p.name.startswith("evaluation_results")
    )
    discovered = []
    for dataset_dir in dataset_dirs:
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
        sim_files = find_simulation_files(dataset_dir)
        if set(sim_files) == set(RUNS):
            discovered.append((dataset_dir.name, dataset_dir, sim_files))
        else:
            print(f"[skip] {dataset_dir.name}: missing runs {sorted(set(RUNS) - set(sim_files))}")
    if not discovered:
        raise FileNotFoundError("No dataset directory contains all five DPSIM run CSVs (1..5).")
    return discovered

def _frame_digest(df: pd.DataFrame) -> str:
    """Stable diagnostic digest of values/column order after preparation."""
    text = df.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_sample_manifest(path: Path | None) -> dict[tuple[str, int], list[str]]:
    if path is None:
        return {}
    manifest = pd.read_csv(path)
    required = {"dataset", "run_num", "caseid"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Sample manifest is missing columns: {sorted(missing)}")
    result: dict[tuple[str, int], list[str]] = {}
    for (dataset, run_num), group in manifest.groupby(["dataset", "run_num"], sort=False):
        result[(str(dataset), int(run_num))] = group["caseid"].astype(str).tolist()
    return result


def _canonical_caseid(value: object) -> str:
    value = str(value)
    return value if value.startswith("Case") else "Case" + value


@contextlib.contextmanager
def capture_or_force_random_sample(forced_caseids: list[str] | None = None):
    """Capture SimilarityEvaluator's random.sample, or force a known case subset.

    The evaluator calls random.sample on a list of trace dictionaries after
    reformatting the real log.  For the three target metrics, forcing the exact
    subset is sufficient; the sampled order does not change the mean optimal
    matching result or EMD.
    """
    original_sample = random.sample
    captured: list[str] = []
    forced = [_canonical_caseid(x) for x in forced_caseids] if forced_caseids else None

    def wrapped(population, k):
        nonlocal captured
        is_trace_population = (
            isinstance(population, list)
            and population
            and isinstance(population[0], dict)
            and "caseid" in population[0]
            and "profile" in population[0]
        )
        if not is_trace_population:
            return original_sample(population, k)

        if forced is not None:
            lookup = {_canonical_caseid(x["caseid"]): x for x in population}
            missing = [cid for cid in forced if cid not in lookup]
            if missing:
                raise ValueError(f"Forced sample contains case IDs not present in test log: {missing[:10]}")
            if len(forced) != k:
                raise ValueError(
                    f"Forced sample has {len(forced)} cases but evaluator requests {k}. "
                    "The manifest must come from the matching original run."
                )
            selected = [lookup[cid] for cid in forced]
        else:
            selected = original_sample(population, k)

        captured[:] = [str(x["caseid"]) for x in selected]
        return selected

    random.sample = wrapped
    try:
        yield captured
    finally:
        random.sample = original_sample


def evaluate_one_exact(
    ds,
    dataset: str,
    run_num: int,
    test_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    forced_caseids: list[str] | None,
) -> tuple[dict[str, float], list[str]]:
    """Call the original DeepSimulator._evaluate_logs unchanged."""
    parms = {
        "gl": {
            "read_options": {"one_timestamp": False},
            # Match pipeline.py exactly, including the extra metrics.  We only
            # return the three requested metrics below.
            "sim_metric": "tsd",
            "add_metrics": ["day_hour_emd", "log_mae", "dl", "mae"],
        }
    }

    with capture_or_force_random_sample(forced_caseids) as captured:
        values = ds.DeepSimulator._evaluate_logs(
            parms,
            test_df.copy(deep=True),
            sim_df.copy(deep=True),
            run_num,
        )
        sampled = list(captured)

    by_metric = {row["metric"]: float(row["sim_val"]) for row in values}
    missing = [m for m in TARGET_METRICS if m not in by_metric]
    if missing:
        raise RuntimeError(f"Original evaluation path did not produce metrics: {missing}")
    return {m: by_metric[m] for m in TARGET_METRICS}, sampled


def build_metric_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in detail.groupby("dataset", sort=True):
        for metric in TARGET_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row = {
                "dataset": dataset,
                "metric": metric,
                **{f"run{r}": _value_for_run(group, metric, r) for r in RUNS},
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "min": values.min(),
                "max": values.max(),
                "count": int(values.notna().sum()),
                "direction": "lower_is_better" if metric in {"day_hour_emd", "mae"} else "higher_is_better",
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _value_for_run(group: pd.DataFrame, metric: str, run: int) -> float:
    values = group.loc[group["run_num"] == run, metric]
    return float("nan") if values.empty else float(values.iloc[0])


def build_dataset_table(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in detail.groupby("dataset", sort=True):
        row = {
            "dataset": dataset,
            "test_file": group["test_file"].iloc[0],
            "test_cases": int(group["test_cases"].iloc[0]),
        }
        for metric in TARGET_METRICS:
            run_values = []
            for run in RUNS:
                value = _value_for_run(group, metric, run)
                row[f"{metric}_run{run}"] = value
                run_values.append(value)
            row[f"{metric}_mean"] = float(np.nanmean(run_values))
        rows.append(row)
    columns = ["dataset", "test_file", "test_cases"]
    for metric in TARGET_METRICS:
        columns.extend([f"{metric}_run{r}" for r in RUNS])
        columns.append(f"{metric}_mean")
    return pd.DataFrame(rows)[columns]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved DPSIM simulation logs through the original DeepSimulator evaluation path."
    )
    parser.add_argument("--dpsim-root", type=Path, default=DEFAULT_DPSIM_ROOT)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=None,
        help=(
            "Optional CSV with dataset,run_num,caseid captured from the original run. "
            "Providing it is the reliable way to reproduce historical metrics exactly "
            "when the original random state is no longer available."
        ),
    )
    parser.add_argument(
        "--seed-before-eval",
        type=int,
        default=None,
        help=(
            "Optional deterministic seed applied immediately before each evaluation. "
            "Do NOT use this to claim equality with an old original run unless that run "
            "used the same random state at evaluator construction."
        ),
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    ds = import_original_deep_simulator()
    forced_samples = load_sample_manifest(args.sample_manifest)
    datasets = discover_datasets(args.dpsim_root, args.datasets)

    print(f"[discover] {len(datasets)} dataset(s)")
    if args.sample_manifest:
        print(f"[repro] forcing sampled test cases from: {args.sample_manifest}")
    elif args.seed_before_eval is not None:
        print(f"[repro] reseeding immediately before evaluation: {args.seed_before_eval}")
    else:
        print(
            "[repro warning] no sample manifest/random-state replay supplied. "
            "The data/evaluation path is original-compatible, but historical exact "
            "equality is not guaranteed because SimilarityEvaluator uses random.sample()."
        )

    all_rows = []
    sample_rows = []
    errors = []

    for dataset, dataset_dir, sim_files in datasets:
        print("\n" + "#" * 96)
        print(f"[dataset] {dataset}")
        try:
            test_path = find_test_log(dataset, args.test_root)
            test_df = read_test_log(test_path, one_timestamp=False)
            # self.log_test contains virtual Start/End, so count real case IDs only.
            test_cases = int(test_df["caseid"].nunique())
            print(f"[test] rows={len(test_df)}, cases={test_cases}, digest={_frame_digest(test_df)}")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] test preparation failed: {message}")
            errors.append({"dataset": dataset, "run_num": "ALL", "file": "", "error": message})
            if args.fail_fast:
                raise
            continue

        for run_num in RUNS:
            sim_path = sim_files[run_num]
            print("\n" + "=" * 96)
            print(f"[run {run_num}] {sim_path.name}")
            try:
                sim_df = read_simulation_csv_like_exported_event_log(sim_path, one_timestamp=False)
                sim_cases = int(sim_df["caseid"].nunique())
                print(f"[simulation] rows={len(sim_df)}, cases={sim_cases}, digest={_frame_digest(sim_df)}")

                if args.seed_before_eval is not None and args.sample_manifest is None:
                    random.seed(args.seed_before_eval)
                    np.random.seed(args.seed_before_eval)

                forced = forced_samples.get((dataset, run_num))
                metrics, sampled_caseids = evaluate_one_exact(
                    ds, dataset, run_num, test_df, sim_df, forced
                )

                for cid in sampled_caseids:
                    sample_rows.append({"dataset": dataset, "run_num": run_num, "caseid": cid})

                row = {
                    "dataset": dataset,
                    "run_num": run_num,
                    "simulation_file": sim_path.name,
                    "simulation_path": str(sim_path),
                    "simulation_rows": len(sim_df),
                    "simulation_cases": sim_cases,
                    "test_file": str(test_path),
                    "test_rows_before_eval": len(test_df),
                    "test_cases": test_cases,
                    "sampled_test_cases": len(sampled_caseids),
                    "test_digest": _frame_digest(test_df),
                    "simulation_digest": _frame_digest(sim_df),
                    **metrics,
                }
                all_rows.append(row)
                print(
                    "[result] "
                    f"day_hour_emd={metrics['day_hour_emd']:.12g}, "
                    f"mae={metrics['mae']:.12g}, dl={metrics['dl']:.12g}, "
                    f"sampled_cases={len(sampled_caseids)}"
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print(f"[ERROR] {dataset} run {run_num}: {message}")
                traceback.print_exc()
                errors.append({"dataset": dataset, "run_num": run_num, "file": str(sim_path), "error": message})
                if args.fail_fast:
                    raise

    if not all_rows:
        print("\nNo evaluation run succeeded.")
        return 1

    detail = pd.DataFrame(all_rows).sort_values(["dataset", "run_num"]).reset_index(drop=True)
    metric_summary = build_metric_summary(detail)
    dataset_table = build_dataset_table(detail)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "evaluation_all_runs.csv"
    dataset_table_path = args.output_dir / "evaluation_dataset_table.csv"
    metric_summary_path = args.output_dir / "evaluation_metric_summary.csv"
    sample_path = args.output_dir / "evaluation_sample_manifest.csv"
    errors_path = args.output_dir / "evaluation_errors.csv"

    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    dataset_table.to_csv(dataset_table_path, index=False, encoding="utf-8-sig")
    metric_summary.to_csv(metric_summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(sample_rows, columns=["dataset", "run_num", "caseid"]).to_csv(
        sample_path, index=False, encoding="utf-8-sig"
    )
    if errors:
        pd.DataFrame(errors).to_csv(errors_path, index=False, encoding="utf-8-sig")

    print("\n" + "#" * 96)
    print("[dataset comparison table: 5 runs + mean]")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(dataset_table.to_string(index=False))

    print("\n[saved]")
    print(f"  Per-run detail : {detail_path}")
    print(f"  Main table     : {dataset_table_path}")
    print(f"  Metric summary : {metric_summary_path}")
    print(f"  Sample manifest: {sample_path}")
    if errors:
        print(f"  Errors         : {errors_path}")

    print("\nMetric direction: day_hour_emd ↓, mae ↓, dl ↑")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
