"""Render a `ComparisonReport` as a Markdown document.

Produces side-by-side tables for perf (per workload+concurrency) and
quality (per task), plus a top-line "regression detected" banner. Pure
rendering — caller passes in the report and a target path.
"""

from __future__ import annotations

from pathlib import Path

from lmbench.compare import ComparisonReport, MetricDelta


def _fmt_num(v: float) -> str:
    if v == 0.0:
        return "0"
    if abs(v) < 1e-3:
        return f"{v:.2e}"
    return f"{v:.4f}"


def _fmt_pct(rel: float) -> str:
    return f"{rel * 100:+.2f}%"


def _flag(d: MetricDelta) -> str:
    return ":warning:" if d.regression else ""


def _render_metric_row(d: MetricDelta) -> str:
    return (
        f"| {d.name} | {_fmt_num(d.baseline)} | {_fmt_num(d.candidate)} | "
        f"{_fmt_num(d.abs_delta)} | {_fmt_pct(d.rel_delta)} | {_flag(d)} |"
    )


def _render_table(deltas: tuple[MetricDelta, ...]) -> str:
    if not deltas:
        return "_(no metrics)_"
    header = "| Metric | Baseline | Candidate | Dabs | Drel | Flag |"
    sep = "|---|---|---|---|---|---|"
    rows = [_render_metric_row(d) for d in deltas]
    return "\n".join([header, sep, *rows])


def render_markdown(
    report: ComparisonReport,
    *,
    title: str = "lmbench: baseline vs candidate",
) -> str:
    """Return the full Markdown text for a comparison report."""
    out: list[str] = [f"# {title}", ""]
    if report.any_regression:
        out += [
            "> :warning: **Regression detected.** "
            "At least one metric exceeds its threshold.",
            "",
        ]
    else:
        out += ["> :white_check_mark: No regressions above threshold.", ""]

    if report.perf:
        out += ["## Performance", ""]
        for cell in report.perf:
            heading = (
                f"### Workload `{cell.workload_name}` @ concurrency={cell.concurrency}"
            )
            out += [heading, "", _render_table(cell.deltas), ""]

    if report.quality:
        out += ["## Quality", ""]
        for q in report.quality:
            out += [f"### Suite `{q.suite_name}`", "", _render_table(q.deltas), ""]

    return "\n".join(out).rstrip() + "\n"


def write_markdown(
    report: ComparisonReport, output_path: Path, **kwargs: str
) -> Path:
    """Render and write a Markdown report; return the written path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(report, **kwargs), encoding="utf-8")
    return output_path
