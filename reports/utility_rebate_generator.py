"""
CooledAI Utility Rebate Generator - Audit-ready PDF report.

Takes 30-day data from BaselineReporter and FinancialMetrics (and shadow_logs
for verification) and produces a professional PDF for utility rebate submission.

Includes Shadow Comparison engine: tracks actual energy vs theoretical minimum
from the RL agent, and outputs charts for Cumulative Energy Waste and Carbon
Offset Potential — primary sales tool for CooledAI.
"""

import io
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default CO2 factor for carbon offset (metric tonnes per MWh = kg/MWh / 1000)
DEFAULT_CO2_KG_PER_KWH = 0.4
REPORT_DAYS = 30


# --- Shadow Comparison Engine ---


def _parse_ts(s: str) -> Optional[datetime]:
    """Parse ISO timestamp from shadow log."""
    if not s:
        return None
    s = str(s).rstrip("Z").replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


@dataclass
class ShadowComparisonResult:
    """Result of Shadow Comparison: actual vs theoretical minimum from RL agent."""

    timestamps: List[datetime]
    cumulative_waste_kwh: List[float]
    carbon_offset_tonnes: List[float]
    total_waste_kwh: float
    total_carbon_offset_tonnes: float
    sample_count: int
    start_ts: datetime
    end_ts: datetime
    avg_actual_cooling_kw: float
    avg_theoretical_cooling_kw: float


class ShadowComparisonEngine:
    """
    Tracks actual energy used by the facility vs the Theoretical Minimum
    calculated by the RL agent (from shadow_logs: proposed_cooling_power_kw).

    Outputs time-series for:
    - Cumulative Energy Waste (kWh)
    - Carbon Offset Potential (tonnes CO2)
    """

    def __init__(
        self,
        shadow_db_path: Optional[Path] = None,
        co2_kg_per_kwh: float = DEFAULT_CO2_KG_PER_KWH,
    ):
        self.shadow_db_path = shadow_db_path or Path("data/shadow_logs.db")
        self.co2_kg_per_kwh = co2_kg_per_kwh

    def run(
        self,
        start_ts: Optional[datetime] = None,
        end_ts: Optional[datetime] = None,
        days: int = REPORT_DAYS,
    ) -> ShadowComparisonResult:
        """
        Run Shadow Comparison over the period.

        For each shadow log sample: waste = (actual_cooling_kw - proposed_cooling_kw) * delta_hours.
        Cumulative waste = running sum. Carbon = cumulative_waste * co2_kg_per_kwh / 1000.
        """
        if end_ts is None:
            end_ts = datetime.utcnow()
        if start_ts is None:
            start_ts = end_ts - timedelta(days=days)

        try:
            from backend.shadow.shadow_logs import get_shadow_logs
            rows = get_shadow_logs(
                start_ts=start_ts,
                end_ts=end_ts,
                limit=100000,
                db_path=self.shadow_db_path,
            )
        except Exception as e:
            logger.warning("ShadowComparisonEngine: get_shadow_logs failed: %s", e)
            rows = []

        if not rows:
            return ShadowComparisonResult(
                timestamps=[],
                cumulative_waste_kwh=[],
                carbon_offset_tonnes=[],
                total_waste_kwh=0.0,
                total_carbon_offset_tonnes=0.0,
                sample_count=0,
                start_ts=start_ts,
                end_ts=end_ts,
                avg_actual_cooling_kw=0.0,
                avg_theoretical_cooling_kw=0.0,
            )

        timestamps: List[datetime] = []
        waste_kwh_per_sample: List[float] = []
        actual_kw_list: List[float] = []
        theoretical_kw_list: List[float] = []

        prev_ts: Optional[datetime] = None
        for r in rows:
            ts_str = r.get("created_at")
            ts = _parse_ts(ts_str) if ts_str else None
            if ts is None:
                continue

            actual_kw = float(r.get("cooling_power_kw") or 0.0)
            theoretical_kw = float(r.get("proposed_cooling_power_kw") or 0.0)

            delta_hours = 0.0
            if prev_ts is not None:
                delta_sec = (ts - prev_ts).total_seconds()
                delta_hours = max(0.0, delta_sec / 3600.0)
            else:
                delta_hours = 5.0 / 60.0

            waste_kw = max(0.0, actual_kw - theoretical_kw)
            waste_kwh = waste_kw * delta_hours

            timestamps.append(ts)
            waste_kwh_per_sample.append(waste_kwh)
            actual_kw_list.append(actual_kw)
            theoretical_kw_list.append(theoretical_kw)
            prev_ts = ts

        cumsum = 0.0
        cumulative_waste_kwh: List[float] = []
        carbon_offset_tonnes: List[float] = []
        for w in waste_kwh_per_sample:
            cumsum += w
            cumulative_waste_kwh.append(cumsum)
            carbon_offset_tonnes.append(cumsum * self.co2_kg_per_kwh / 1000.0)

        n = len(timestamps)
        avg_actual = sum(actual_kw_list) / n if actual_kw_list else 0.0
        avg_theoretical = sum(theoretical_kw_list) / n if theoretical_kw_list else 0.0

        return ShadowComparisonResult(
            timestamps=timestamps,
            cumulative_waste_kwh=cumulative_waste_kwh,
            carbon_offset_tonnes=carbon_offset_tonnes,
            total_waste_kwh=cumsum,
            total_carbon_offset_tonnes=cumsum * self.co2_kg_per_kwh / 1000.0,
            sample_count=n,
            start_ts=start_ts,
            end_ts=end_ts,
            avg_actual_cooling_kw=avg_actual,
            avg_theoretical_cooling_kw=avg_theoretical,
        )


def _build_shadow_comparison_charts(
    result: ShadowComparisonResult,
    width_inches: float = 6.0,
    height_inches: float = 3.0,
    dpi: int = 100,
) -> Tuple[Optional[bytes], Optional[bytes]]:
    """Build both chart images. Returns (cumulative_waste_png, carbon_offset_png)."""
    chart1 = _build_chart_image("cumulative_waste", result, width_inches, height_inches, dpi)
    chart2 = _build_chart_image("carbon_offset", result, width_inches, height_inches, dpi)
    return chart1, chart2


def _build_chart_image(
    chart_type: str,
    result: ShadowComparisonResult,
    width_inches: float = 6.0,
    height_inches: float = 3.0,
    dpi: int = 100,
) -> Optional[bytes]:
    """Build matplotlib chart and return PNG bytes."""
    if not result.timestamps or not result.cumulative_waste_kwh:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not installed; Shadow Comparison charts skipped")
        return None

    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    x = result.timestamps
    if chart_type == "cumulative_waste":
        y = result.cumulative_waste_kwh
        ax.set_ylabel("Cumulative Energy Waste (kWh)", fontsize=10)
        ax.set_title("Cumulative Energy Waste — Actual vs Theoretical Minimum", fontsize=11, fontweight="bold")
        ax.fill_between(x, y, alpha=0.4, color="#e74c3c")
    else:
        y = result.carbon_offset_tonnes
        ax.set_ylabel("Carbon Offset Potential (tCO₂)", fontsize=10)
        ax.set_title("Carbon Offset Potential — Avoidable Emissions", fontsize=11, fontweight="bold")
        ax.fill_between(x, y, alpha=0.4, color="#27ae60")

    ax.plot(x, y, color="#2c3e50" if chart_type == "cumulative_waste" else "#1e8449", linewidth=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=15)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close()
    buf.seek(0)
    return buf.read()


def _get_30day_baseline_data(
    baseline_db_path: Path,
    shadow_db_path: Path,
    end_ts: datetime,
    utility_rate: float,
    co2_kg_per_kwh: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Fetch 30-day baseline comparison and shadow logs for verification."""
    start_ts = end_ts - timedelta(days=REPORT_DAYS)
    baseline_report = None
    try:
        from backend.baseline.baseline_reporter import BaselineReporter
        reporter = BaselineReporter(
            db_path=baseline_db_path,
            utility_rate_per_kwh=utility_rate,
            co2_kg_per_kwh=co2_kg_per_kwh,
        )
        baseline_report = reporter.generate_comparison_report(
            start_ts=start_ts, end_ts=end_ts, hours=REPORT_DAYS * 24.0
        )
    except Exception as e:
        logger.warning("Baseline report failed: %s", e)

    verification_log = []
    try:
        from backend.shadow.shadow_logs import get_shadow_logs
        rows = get_shadow_logs(start_ts=start_ts, end_ts=end_ts, limit=50000, db_path=shadow_db_path)
        for r in rows:
            verification_log.append({
                "created_at": r.get("created_at"),
                "recommended_cooling_delta": r.get("recommended_cooling_delta"),
                "actual_pue": r.get("actual_pue"),
                "proposed_pue": r.get("proposed_pue"),
                "it_power_kw": r.get("it_power_kw"),
                "cooling_power_kw": r.get("cooling_power_kw"),
                "proposed_cooling_power_kw": r.get("proposed_cooling_power_kw"),
            })
    except Exception as e:
        logger.warning("Shadow logs for verification failed: %s", e)

    report_dict = None
    if baseline_report is not None:
        report_dict = {
            "current_pue": baseline_report.current_pue,
            "projected_cooledai_pue": baseline_report.projected_cooledai_pue,
            "pue_improvement_pct": baseline_report.pue_improvement_pct,
            "kwh_saved": baseline_report.kwh_saved_24h,
            "mwh_saved": baseline_report.kwh_saved_24h / 1000.0,
            "co2_kg": baseline_report.co2_reduction_kg,
            "co2_tonnes": baseline_report.co2_reduction_tonnes,
            "dollars_saved": baseline_report.dollars_saved_24h,
            "sample_count": baseline_report.sample_count,
            "start_ts": baseline_report.start_ts,
            "end_ts": baseline_report.end_ts,
        }
    else:
        report_dict = {
            "current_pue": None,
            "projected_cooledai_pue": None,
            "pue_improvement_pct": 0.0,
            "kwh_saved": 0.0,
            "mwh_saved": 0.0,
            "co2_kg": 0.0,
            "co2_tonnes": 0.0,
            "dollars_saved": 0.0,
            "sample_count": 0,
            "start_ts": start_ts,
            "end_ts": end_ts,
        }

    return report_dict, verification_log


class UtilityRebateGenerator:
    """
    Generates an audit-ready PDF report for utility rebate submission using
    30-day BaselineReporter and FinancialMetrics data, plus verification log
    from shadow_logs (Active Demand Management).
    """

    def __init__(
        self,
        baseline_db_path: Optional[Path] = None,
        shadow_db_path: Optional[Path] = None,
        utility_rate: float = 0.12,
        co2_kg_per_kwh: float = DEFAULT_CO2_KG_PER_KWH,
    ):
        self.baseline_db_path = baseline_db_path or Path("data/baseline.db")
        self.shadow_db_path = shadow_db_path or Path("data/shadow_logs.db")
        self.utility_rate = utility_rate
        self.co2_kg_per_kwh = co2_kg_per_kwh

    def generate_pdf(self, end_ts: Optional[datetime] = None) -> bytes:
        """Build the PDF and return bytes."""
        end_ts = end_ts or datetime.utcnow()
        data, verification_log = _get_30day_baseline_data(
            self.baseline_db_path,
            self.shadow_db_path,
            end_ts,
            self.utility_rate,
            self.co2_kg_per_kwh,
        )

        buffer = io.BytesIO()
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
            )
        except ImportError:
            raise RuntimeError("reportlab is required for PDF reports. Install with: pip install reportlab")

        try:
            from reportlab.platypus import Image
        except ImportError:
            Image = None

        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch,
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            name="ReportTitle",
            parent=styles["Heading1"],
            fontSize=16,
            spaceAfter=12,
        )
        heading_style = styles["Heading2"]
        body = styles["Normal"]
        elements = []

        start_ts = data["start_ts"]

        # --- Shadow Comparison (Primary Sales Tool) ---
        shadow_engine = ShadowComparisonEngine(
            shadow_db_path=self.shadow_db_path,
            co2_kg_per_kwh=self.co2_kg_per_kwh,
        )
        shadow_result = shadow_engine.run(start_ts=start_ts, end_ts=data["end_ts"], days=REPORT_DAYS)

        if shadow_result.sample_count > 0:
            elements.append(Paragraph("Shadow Comparison — Actual vs Theoretical Minimum", title_style))
            elements.append(Paragraph(
                "Comparison of facility actual energy use vs the Theoretical Minimum calculated by our RL agent. "
                "Energy waste represents the gap between reactive control and proactive optimization.",
                body,
            ))
            elements.append(Spacer(1, 0.2 * inch))

            summary_data = [
                ["Metric", "Value"],
                ["Total Energy Waste (kWh)", f"{shadow_result.total_waste_kwh:,.1f}"],
                ["Carbon Offset Potential (tCO₂)", f"{shadow_result.total_carbon_offset_tonnes:,.2f}"],
                ["Optimization Samples", f"{shadow_result.sample_count:,}"],
                ["Avg Actual Cooling (kW)", f"{shadow_result.avg_actual_cooling_kw:.1f}"],
                ["Avg Theoretical Minimum (kW)", f"{shadow_result.avg_theoretical_cooling_kw:.1f}"],
            ]
            t_shadow = Table(summary_data, colWidths=[3.0 * inch, 2.5 * inch])
            t_shadow.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#27ae60")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#E8F8F5")),
            ]))
            elements.append(t_shadow)
            elements.append(Spacer(1, 0.3 * inch))

            chart1_png, chart2_png = _build_shadow_comparison_charts(shadow_result, width_inches=6.0, height_inches=3.0)
            if chart1_png and Image is not None:
                elements.append(Paragraph("1. Cumulative Energy Waste (kWh)", heading_style))
                img1 = Image(io.BytesIO(chart1_png), width=6 * inch, height=3 * inch)
                elements.append(img1)
                elements.append(Spacer(1, 0.25 * inch))
            if chart2_png and Image is not None:
                elements.append(Paragraph("2. Carbon Offset Potential (tCO₂)", heading_style))
                img2 = Image(io.BytesIO(chart2_png), width=6 * inch, height=3 * inch)
                elements.append(img2)
            elements.append(Spacer(1, 0.4 * inch))
            elements.append(PageBreak())

        else:
            elements.append(Paragraph("Shadow Comparison — Actual vs Theoretical Minimum", title_style))
            elements.append(Paragraph(
                "No shadow log data in this period. Enable Shadow Mode and run the optimization pipeline "
                "to collect Actual vs Theoretical Minimum data for Cumulative Energy Waste and Carbon Offset charts.",
                body,
            ))
            elements.append(Spacer(1, 0.3 * inch))

        end_ts_str = data["end_ts"].strftime("%Y-%m-%d %H:%M UTC") if hasattr(data["end_ts"], "strftime") else str(data["end_ts"])
        start_ts_str = start_ts.strftime("%Y-%m-%d %H:%M UTC") if hasattr(start_ts, "strftime") else str(start_ts)

        elements.append(Paragraph("Utility Rebate — Demand Management Verification Report", title_style))
        elements.append(Paragraph(f"Report Period: {start_ts_str} to {end_ts_str} ({REPORT_DAYS} days)", body))
        elements.append(Paragraph("CooledAI — Active Demand Management", body))
        elements.append(Spacer(1, 0.25 * inch))

        # 1. Baseline vs. Optimized PUE (Monthly Average)
        elements.append(Paragraph("1. Baseline vs. Optimized PUE (Monthly Average)", heading_style))
        current_pue = data["current_pue"]
        optimized_pue = data["projected_cooledai_pue"]
        if current_pue is not None and optimized_pue is not None:
            pue_data = [
                ["Metric", "Value"],
                ["Baseline PUE (monthly average)", f"{current_pue:.4f}"],
                ["Optimized PUE (CooledAI projected, monthly average)", f"{optimized_pue:.4f}"],
                ["PUE Improvement", f"{data['pue_improvement_pct']:.2f}%"],
            ]
        else:
            pue_data = [
                ["Metric", "Value"],
                ["Baseline PUE (monthly average)", "No baseline data in period"],
                ["Optimized PUE (CooledAI projected)", "—"],
                ["PUE Improvement", "—"],
            ]
        t1 = Table(pue_data, colWidths=[3.5 * inch, 2.5 * inch])
        t1.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#E7E6E6")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(t1)
        elements.append(Spacer(1, 0.3 * inch))

        # 2. Total MWh Saved (Total and per-cooling-unit)
        elements.append(Paragraph("2. Total MWh Saved", heading_style))
        mwh = data["mwh_saved"]
        kwh = data["kwh_saved"]
        mwh_data = [
            ["Source", "MWh Saved", "kWh Saved"],
            ["Facility (aggregate)", f"{mwh:.4f}", f"{kwh:.2f}"],
        ]
        elements.append(Paragraph(
            "Per-cooling-unit breakdown requires unit-level telemetry; this report shows facility aggregate.",
            body,
        ))
        t2 = Table(mwh_data, colWidths=[2.5 * inch, 1.5 * inch, 1.5 * inch])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#E7E6E6")),
        ]))
        elements.append(t2)
        elements.append(Spacer(1, 0.3 * inch))

        # 3. Estimated Carbon Offset (Metric Tons CO2)
        elements.append(Paragraph("3. Estimated Carbon Offset", heading_style))
        tonnes = data["co2_tonnes"]
        kg = data["co2_kg"]
        carbon_data = [
            ["Metric", "Value"],
            ["Carbon offset (metric tonnes CO₂)", f"{tonnes:.4f} tCO₂"],
            ["Carbon offset (kg CO₂)", f"{kg:.2f} kg"],
            ["Emission factor used", f"{self.co2_kg_per_kwh} kg CO₂/kWh"],
        ]
        t3 = Table(carbon_data, colWidths=[3.5 * inch, 2.5 * inch])
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#E7E6E6")),
        ]))
        elements.append(t3)
        elements.append(Spacer(1, 0.3 * inch))

        # 4. Verification Log: timestamped optimization events (Active Demand Management)
        elements.append(Paragraph("4. Verification Log — Active Demand Management", heading_style))
        elements.append(Paragraph(
            "Timestamped record of optimization events demonstrating active demand management.",
            body,
        ))
        if verification_log:
            log_rows = [
                ["Timestamp (UTC)", "Cooling Delta", "Actual PUE", "Proposed PUE", "IT Power (kW)", "Description"],
            ]
            for v in verification_log[:500]:  # cap for PDF size
                ts = v.get("created_at") or ""
                if len(ts) > 19:
                    ts = ts[:19].replace("T", " ")
                delta = v.get("recommended_cooling_delta")
                actual = v.get("actual_pue")
                proposed = v.get("proposed_pue")
                it_kw = v.get("it_power_kw")
                desc = "Cooling adjustment applied"
                log_rows.append([
                    ts,
                    f"{delta:.3f}" if delta is not None else "—",
                    f"{actual:.3f}" if actual is not None else "—",
                    f"{proposed:.3f}" if proposed is not None else "—",
                    f"{it_kw:.1f}" if it_kw is not None else "—",
                    desc,
                ])
            if len(verification_log) > 500:
                log_rows.append(["(showing first 500 events)", "", "", "", "", f"Total events: {len(verification_log)}"])
            t4 = Table(log_rows, colWidths=[1.2 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch, 1.5 * inch])
            t4.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F2F2F2")),
            ]))
            elements.append(t4)
        else:
            elements.append(Paragraph("No optimization events recorded in this period (shadow logs empty or not enabled).", body))

        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(
            f"Report generated at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC. "
            "This document is intended for utility rebate verification and audit.",
            ParagraphStyle(name="Footer", parent=body, fontSize=8, textColor=colors.grey),
        ))

        doc.build(elements)
        buffer.seek(0)
        return buffer.read()


def generate_utility_rebate_pdf(
    end_ts: Optional[datetime] = None,
    baseline_db_path: Optional[Path] = None,
    shadow_db_path: Optional[Path] = None,
    utility_rate: float = 0.12,
    co2_kg_per_kwh: float = DEFAULT_CO2_KG_PER_KWH,
) -> bytes:
    """Convenience: generate the utility rebate PDF for the last 30 days."""
    generator = UtilityRebateGenerator(
        baseline_db_path=baseline_db_path,
        shadow_db_path=shadow_db_path,
        utility_rate=utility_rate,
        co2_kg_per_kwh=co2_kg_per_kwh,
    )
    return generator.generate_pdf(end_ts=end_ts)
