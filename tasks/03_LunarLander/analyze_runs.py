from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy import stats

from eval_export import EVAL_FINAL_FILENAME, load_eval_final

PRIMARY_METRIC_KEY = "success_rate"

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

METHODS_NOTE = """\
### Methods (multiseed summary)
- Each seed is one independent training run; **eval_final.json** is deterministic eval on **ac_best.pt** when present (`eval_source: best_checkpoint`; `env_step` is the step when that checkpoint was saved). Legacy runs may lack `eval_source`.
- **95% CI**: one-sample t interval for the mean across seeds (t on seed means).
- **Paired comparison** (``--compare``): only seeds present in both groups; **paired two-sided t-test** on differences (A−B).
- **95% CI for mean difference**: t interval on paired differences (same as one-sample t on diffs).
- **d_z**: paired Cohen's d_z = mean(diff) / SD(diff).
- **Holm** (``--holm``): step-down Holm–Bonferroni on two-sided p-values across listed metrics (exploratory; pre-specify **success_rate** as primary).
"""


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


@dataclass(frozen=True)
class PairedTestResult:
    n: int
    mean_diff: float
    std_diff: float
    se_diff: float
    ci_low: float
    ci_high: float
    t_stat: float
    p_value: float
    cohens_dz: float


def _t_crit_two_sided(alpha: float, df: int) -> float:
    if df <= 0:
        return float("nan")
    if alpha != 0.05:
        return _t_crit_binary_search(alpha, df)
    if df <= len(_T_CRIT_95):
        return _T_CRIT_95[df - 1]
    return 1.96


def _t_crit_binary_search(alpha: float, df: int) -> float:
    return float(stats.t.ppf(1.0 - alpha / 2.0, df))


def _p_value_two_sided_t(t_stat: float, df: int) -> float:
    if df <= 0 or math.isnan(t_stat):
        return float("nan")
    return float(2.0 * stats.t.sf(abs(t_stat), df))


def _summary(values: list[float], *, alpha: float = 0.05) -> SummaryStats | None:
    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    if n == 1:
        return SummaryStats(n=n, mean=mean, std=0.0, ci_low=mean, ci_high=mean)
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    margin = _t_crit_two_sided(alpha, n - 1) * std / math.sqrt(n)
    return SummaryStats(
        n=n,
        mean=mean,
        std=std,
        ci_low=mean - margin,
        ci_high=mean + margin,
    )


def _paired_test(a: list[float], b: list[float], *, alpha: float = 0.05) -> PairedTestResult | None:
    if len(a) != len(b) or len(a) < 2:
        return None
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    std_d = math.sqrt(var_d)
    se_d = std_d / math.sqrt(n) if n > 0 else float("nan")
    if var_d <= 0.0:
        t_stat = float("inf") if mean_d != 0.0 else 0.0
        p_val = 0.0 if mean_d != 0.0 else 1.0
        dz = float("inf") if mean_d != 0.0 else 0.0
        return PairedTestResult(
            n=n,
            mean_diff=mean_d,
            std_diff=std_d,
            se_diff=se_d,
            ci_low=mean_d,
            ci_high=mean_d,
            t_stat=t_stat,
            p_value=p_val,
            cohens_dz=dz,
        )
    t_stat = mean_d / se_d
    df = n - 1
    margin = _t_crit_two_sided(alpha, df) * se_d
    return PairedTestResult(
        n=n,
        mean_diff=mean_d,
        std_diff=std_d,
        se_diff=se_d,
        ci_low=mean_d - margin,
        ci_high=mean_d + margin,
        t_stat=t_stat,
        p_value=_p_value_two_sided_t(t_stat, df),
        cohens_dz=mean_d / std_d,
    )


def _holm_adjusted_p(p_values: list[float]) -> list[float]:
    """Step-down Holm–Bonferroni adjusted p-values (preserves order)."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * m
    prev = 0.0
    for rank, (idx, p) in enumerate(indexed, start=1):
        adj = min(1.0, p * (m - rank + 1))
        adj = max(adj, prev)
        adjusted[idx] = adj
        prev = adj
    return adjusted


def _format_p(p: float) -> str:
    if math.isnan(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


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


def _format_summary(stats: SummaryStats | None) -> str:
    if stats is None:
        return "—"
    return (
        f"{stats.mean:.4f} ± {stats.std:.4f} "
        f"[{stats.ci_low:.4f}, {stats.ci_high:.4f}] (n={stats.n})"
    )


def _markdown_table(run_group_dir: Path, records: list[SeedRecord]) -> str:
    seeds = sorted(r.seed for r in records)
    lines = [
        f"## {run_group_dir}",
        "",
        f"Seeds with {EVAL_FINAL_FILENAME}: {len(records)} ({', '.join(str(s) for s in seeds)})",
        "",
        "| metric | mean ± std [95% CI] |",
        "|--------|---------------------|",
    ]
    for metric_key, label in METRIC_COLUMNS:
        stats = _summary(_collect_metric(records, metric_key))
        tag = " **(primary)**" if metric_key == PRIMARY_METRIC_KEY else ""
        lines.append(f"| {label}{tag} | {_format_summary(stats)} |")
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


def _paired_values_for_metric(
    common_seeds: list[int],
    left_by_seed: dict[int, SeedRecord],
    right_by_seed: dict[int, SeedRecord],
    metric_key: str,
) -> tuple[list[float], list[float]] | None:
    a_vals: list[float] = []
    b_vals: list[float] = []
    for seed in common_seeds:
        av = _metric_value(left_by_seed[seed].metrics, metric_key)
        bv = _metric_value(right_by_seed[seed].metrics, metric_key)
        if av is None or bv is None:
            return None
        a_vals.append(av)
        b_vals.append(bv)
    return a_vals, b_vals


def _compare_section(
    left_dir: Path,
    left_records: list[SeedRecord],
    right_dir: Path,
    right_records: list[SeedRecord],
    *,
    holm: bool,
) -> str:
    left_by_seed = {r.seed: r for r in left_records}
    right_by_seed = {r.seed: r for r in right_records}
    left_seeds = set(left_by_seed)
    right_seeds = set(right_by_seed)
    common = sorted(left_seeds & right_seeds)
    only_a = sorted(left_seeds - right_seeds)
    only_b = sorted(right_seeds - left_seeds)

    lines = [
        "## Paired comparison (A − B)",
        "",
        f"- A: `{left_dir}`",
        f"- B: `{right_dir}`",
        f"- Paired seeds (n={len(common)}): {', '.join(str(s) for s in common) if common else '—'}",
    ]
    if only_a:
        lines.append(f"- Only in A (excluded): {', '.join(str(s) for s in only_a)}")
    if only_b:
        lines.append(f"- Only in B (excluded): {', '.join(str(s) for s in only_b)}")
    lines.extend(
        [
            "",
            "| metric | mean diff [95% CI] | t | p (two-sided) | d_z |",
            "|--------|-------------------|---|---------------|-----|",
        ]
    )

    results: list[tuple[str, str, PairedTestResult | None]] = []
    p_for_holm: list[float] = []
    holm_keys: list[str] = []

    for metric_key, label in METRIC_COLUMNS:
        paired = _paired_values_for_metric(common, left_by_seed, right_by_seed, metric_key)
        if paired is None:
            results.append((metric_key, label, None))
            continue
        a_vals, b_vals = paired
        test = _paired_test(a_vals, b_vals)
        results.append((metric_key, label, test))
        if test is not None and not math.isnan(test.p_value):
            p_for_holm.append(test.p_value)
            holm_keys.append(metric_key)

    holm_map: dict[str, float] = {}
    if holm and p_for_holm:
        adjusted = _holm_adjusted_p(p_for_holm)
        holm_map = dict(zip(holm_keys, adjusted, strict=True))

    for metric_key, label, test in results:
        primary = metric_key == PRIMARY_METRIC_KEY
        label_cell = f"{label} **(primary)**" if primary else label
        if test is None:
            lines.append(f"| {label_cell} | — | — | — | — |")
            continue
        diff_ci = f"{test.mean_diff:+.4f} [{test.ci_low:+.4f}, {test.ci_high:+.4f}]"
        p_cell = _format_p(test.p_value)
        if holm and metric_key in holm_map:
            p_cell += f" (Holm {_format_p(holm_map[metric_key])})"
        lines.append(
            f"| {label_cell} | {diff_ci} | {test.t_stat:.3f} | {p_cell} | {test.cohens_dz:.3f} |"
        )

    lines.extend(["", METHODS_NOTE.strip()])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate LunarLander multiseed eval_final.json with basic inference.",
    )
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
    parser.add_argument(
        "--holm",
        action="store_true",
        help="Report Holm–Bonferroni adjusted p-values across metrics (compare only).",
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
        print(_compare_section(run_dir, records, other_dir, other_records, holm=args.holm))


if __name__ == "__main__":
    main()
