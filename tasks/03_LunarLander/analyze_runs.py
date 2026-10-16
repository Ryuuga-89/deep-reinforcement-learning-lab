from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_export import EVAL_FINAL_FILENAME, load_eval_final

# two-sided 95% t critical values for df = n - 1 (df 1..30)
_T_CRIT_95: tuple[float, ...] = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)

METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("success_rate", "success_rate"),
    ("mean_episode_return", "mean_return"),
    ("land_success_rate", "land_rate"),
    ("crash_rate", "crash_rate"),
    ("timeout_rate", "timeout_rate"),
    ("mean_main_fuel_on_land", "main_fuel_land"),
    ("mean_side_fuel_on_land", "side_fuel_land"),
    ("mean_abs_tilt_on_land", "abs_tilt_land"),
    ("mean_abs_angular_speed_on_land", "abs_angvel_land"),
)


@dataclass(frozen=True)
class SeedRecord:
    seed: int
    metrics: dict[str, Any]


@dataclass(frozen=True)
class SummaryStats:
    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float


def _t_crit_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df <= len(_T_CRIT_95):
        return _T_CRIT_95[df - 1]
    return 1.96


def _summary(values: list[float]) -> SummaryStats | None:
    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    if n == 1:
        return SummaryStats(n=n, mean=mean, std=0.0, ci_low=mean, ci_high=mean)
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    margin = _t_crit_95(n - 1) * std / math.sqrt(n)
    return SummaryStats(
        n=n,
        mean=mean,
        std=std,
        ci_low=mean - margin,
        ci_high=mean + margin,
    )


def _load_run_group(run_group_dir: Path) -> list[SeedRecord]:
    if not run_group_dir.is_dir():
        raise FileNotFoundError(f"run group directory not found: {run_group_dir}")

    records: list[SeedRecord] = []
    for seed_dir in sorted(run_group_dir.glob("seed_*")):
        eval_path = seed_dir / EVAL_FINAL_FILENAME
        if not eval_path.is_file():
            continue
        payload = load_eval_final(eval_path)
        seed = int(payload["seed"])
        metrics = payload["metrics"]
        if not isinstance(metrics, dict):
            raise ValueError(f"invalid metrics in {eval_path}")
        records.append(SeedRecord(seed=seed, metrics=metrics))
    return records


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def _collect_metric(records: list[SeedRecord], metric_key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _metric_value(record.metrics, metric_key)
        if value is not None:
            values.append(value)
    return values


def _paired_t_test(a: list[float], b: list[float]) -> tuple[float, float, int]:
    if len(a) != len(b) or len(a) < 2:
        return float("nan"), float("nan"), len(a)
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    if var_d <= 0.0:
        return float("inf") if mean_d != 0.0 else 0.0, mean_d, n
    t_stat = mean_d / math.sqrt(var_d / n)
    df = n - 1
    t_crit = _t_crit_95(df)
    p_approx = 0.05 if abs(t_stat) >= t_crit else 1.0
    return t_stat, p_approx, n


def _format_summary(stats: SummaryStats | None) -> str:
    if stats is None:
        return "—"
    return (
        f"{stats.mean:.4f} ± {stats.std:.4f} "
        f"[{stats.ci_low:.4f}, {stats.ci_high:.4f}] (n={stats.n})"
    )


def _markdown_table(run_group_dir: Path, records: list[SeedRecord]) -> str:
    lines = [
        f"## {run_group_dir}",
        "",
        f"Seeds with eval_final.json: {len(records)}",
        "",
        "| metric | summary (mean ± std, 95% CI) |",
        "|--------|------------------------------|",
    ]
    for metric_key, label in METRIC_COLUMNS:
        stats = _summary(_collect_metric(records, metric_key))
        lines.append(f"| {label} | {_format_summary(stats)} |")
    return "\n".join(lines)


def _write_csv(path: Path, records: list[SeedRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = [key for key, _ in METRIC_COLUMNS]
    fieldnames = ["seed"] + metric_keys
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda r: r.seed):
            row: dict[str, Any] = {"seed": record.seed}
            for key in metric_keys:
                value = _metric_value(record.metrics, key)
                row[key] = "" if value is None else value
            writer.writerow(row)


def _compare_section(
    left_dir: Path,
    left_records: list[SeedRecord],
    right_dir: Path,
    right_records: list[SeedRecord],
) -> str:
    left_by_seed = {r.seed: r for r in left_records}
    right_by_seed = {r.seed: r for r in right_records}
    common = sorted(set(left_by_seed) & set(right_by_seed))
    lines = [
        f"## Paired comparison",
        "",
        f"- A: `{left_dir}`",
        f"- B: `{right_dir}`",
        f"- Common seeds: {len(common)}",
        "",
        "| metric | mean diff (A−B) | paired t | approx. significant (α=0.05) |",
        "|--------|-----------------|----------|------------------------------|",
    ]
    for metric_key, label in METRIC_COLUMNS:
        a_vals = []
        b_vals = []
        for seed in common:
            av = _metric_value(left_by_seed[seed].metrics, metric_key)
            bv = _metric_value(right_by_seed[seed].metrics, metric_key)
            if av is None or bv is None:
                continue
            a_vals.append(av)
            b_vals.append(bv)
        if len(a_vals) < 2:
            lines.append(f"| {label} | — | — | — |")
            continue
        diffs = [x - y for x, y in zip(a_vals, b_vals, strict=True)]
        mean_diff = sum(diffs) / len(diffs)
        t_stat, p_approx, _ = _paired_t_test(a_vals, b_vals)
        sig = "yes" if p_approx <= 0.05 else "no"
        lines.append(
            f"| {label} | {mean_diff:+.4f} | {t_stat:.3f} | {sig} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LunarLander multiseed eval_final.json.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run group directory (contains seed_XXXX/ subdirs).",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional second run group for paired comparison.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write per-seed CSV.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    records = _load_run_group(run_dir)
    if not records:
        print(f"No {EVAL_FINAL_FILENAME} found under {run_dir}", file=sys.stderr)
        raise SystemExit(1)

    print(_markdown_table(run_dir, records))
    if args.csv is not None:
        _write_csv(args.csv.resolve(), records)
        print(f"\nWrote CSV: {args.csv.resolve()}", file=sys.stderr)

    if args.compare is not None:
        other_dir = args.compare.resolve()
        other_records = _load_run_group(other_dir)
        if not other_records:
            print(f"No eval files under compare dir: {other_dir}", file=sys.stderr)
            raise SystemExit(1)
        print()
        print(_compare_section(run_dir, records, other_dir, other_records))


if __name__ == "__main__":
    main()
