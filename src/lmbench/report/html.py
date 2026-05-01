"""Render a `ComparisonReport` as an interactive Plotly HTML document.

Falls back to a plain-HTML table if `plotly` is missing — the report
should still be useful in environments where the [gpu] extra wasn't
installed but a baseline summary needs to be eyeballed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lmbench.compare import ComparisonReport, MetricDelta


def _table_rows(deltas: tuple[MetricDelta, ...]) -> list[list[Any]]:
    return [
        [d.name, d.baseline, d.candidate, d.abs_delta, d.rel_delta, d.regression]
        for d in deltas
    ]


_HEADERS = ["Metric", "Baseline", "Candidate", "Dabs", "Drel", "Regression?"]


def _try_import_plotly() -> Any:
    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError:  # pragma: no cover
        return None
    return go, to_html


def _render_plotly(report: ComparisonReport, title: str) -> str:
    plotly = _try_import_plotly()
    if plotly is None:  # pragma: no cover
        return _render_plain_html(report, title)
    go, to_html = plotly
    figs: list[Any] = []
    for cell in report.perf:
        rows = _table_rows(cell.deltas)
        fig = go.Figure(
            data=[
                go.Table(
                    header={"values": _HEADERS},
                    cells={"values": list(map(list, zip(*rows, strict=False)))},
                )
            ]
        )
        fig.update_layout(
            title=f"Perf: {cell.workload_name} @ concurrency={cell.concurrency}"
        )
        figs.append(fig)
    for q in report.quality:
        rows = _table_rows(q.deltas)
        fig = go.Figure(
            data=[
                go.Table(
                    header={"values": _HEADERS},
                    cells={"values": list(map(list, zip(*rows, strict=False)))},
                )
            ]
        )
        fig.update_layout(title=f"Quality: {q.suite_name}")
        figs.append(fig)
    body_parts = [to_html(f, include_plotlyjs="cdn", full_html=False) for f in figs]
    body = "\n".join(body_parts) if body_parts else "<p>(no data)</p>"
    banner = (
        '<p style="color:#b00">Regression detected.</p>'
        if report.any_regression
        else "<p>No regressions above threshold.</p>"
    )
    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title></head><body>"
        f"<h1>{title}</h1>{banner}{body}</body></html>"
    )


def _render_plain_html(report: ComparisonReport, title: str) -> str:
    """Plain-HTML fallback when Plotly is unavailable."""

    def _render_block(name: str, deltas: tuple[MetricDelta, ...]) -> str:
        if not deltas:
            return f"<h3>{name}</h3><p>(no metrics)</p>"
        header_cells = "".join(f"<th>{h}</th>" for h in _HEADERS)
        rows = "".join(
            "<tr>"
            + "".join(
                f"<td>{cell}</td>"
                for cell in (
                    d.name,
                    f"{d.baseline:.4f}",
                    f"{d.candidate:.4f}",
                    f"{d.abs_delta:.4f}",
                    f"{d.rel_delta * 100:+.2f}%",
                    "yes" if d.regression else "",
                )
            )
            + "</tr>"
            for d in deltas
        )
        return (
            f"<h3>{name}</h3><table border='1' cellpadding='4'>"
            f"<thead><tr>{header_cells}</tr></thead><tbody>{rows}</tbody></table>"
        )

    parts: list[str] = []
    for cell in report.perf:
        parts.append(
            _render_block(
                f"Perf: {cell.workload_name} @ {cell.concurrency}", cell.deltas
            )
        )
    for q in report.quality:
        parts.append(_render_block(f"Quality: {q.suite_name}", q.deltas))
    body = "\n".join(parts) if parts else "<p>(no data)</p>"
    banner = (
        '<p style="color:#b00">Regression detected.</p>'
        if report.any_regression
        else "<p>No regressions above threshold.</p>"
    )
    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title></head><body>"
        f"<h1>{title}</h1>{banner}{body}</body></html>"
    )


def render_html(
    report: ComparisonReport,
    *,
    title: str = "lmbench: baseline vs candidate",
) -> str:
    """Return the full HTML text for a comparison report."""
    return _render_plotly(report, title)


def write_html(report: ComparisonReport, output_path: Path, **kwargs: str) -> Path:
    """Render and write an HTML report; return the written path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(report, **kwargs), encoding="utf-8")
    return output_path
