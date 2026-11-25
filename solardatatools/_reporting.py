"""Build and format reports for the current DataHandler attempt."""

from solardatatools._units import POWER_TO_W, unit_scale


PIPELINE_BANNER = """
*********************************************
* Solar Data Tools Data Onboarding Pipeline *
*********************************************

Prepare PV power or irradiance time series for quality checks,
plotting, and further analysis.

Authors: Bennet Meyers and Sara Miskovich, SLAC

This work was supported by the U.S. Department of Energy’s Office
of Critical Minerals and Energy Innovation (CMEI), Integrated
Energy Systems Office (IESO), under Agreement 34368.
"""


def pipeline_report(
    *,
    num_days,
    sampling,
    capacity,
    units,
    diagnostics,
    quality,
    clearness,
    clipping,
    clipped_fraction,
    capacity_change,
    quality_warning,
    time_shifts,
    timezone_advisory,
    outcomes,
):
    capacity_kw = (
        capacity * unit_scale(units, "kW", POWER_TO_W)
        if units in POWER_TO_W and capacity is not None
        else None
    )
    timing = diagnostics or {}
    return {
        "length": num_days / 365 if num_days is not None else None,
        "capacity": capacity_kw,
        "sampling": sampling,
        "timestamp rejection count": timing.get("rejection_count"),
        "timestamp collision count": timing.get("collision_count"),
        "quality score": quality,
        "clearness score": clearness,
        "inverter clipping": clipping,
        "clipped fraction": clipped_fraction,
        "capacity change": capacity_change,
        "data quality warning": quality_warning,
        "time shift correction": time_shifts,
        "time zone correction": 0,
        "possible timezone misalignment": timezone_advisory,
        "stage outcomes": {name: outcome.copy() for name, outcome in outcomes.items()},
    }


def format_outcomes(outcomes):
    lines = []
    for name, outcome in outcomes.items():
        if outcome["reason"] is not None:
            lines.append(f"{name}: {outcome['status']} ({outcome['reason']})")
        if "warning" in outcome:
            lines.append(f"{name}: {outcome['warning']}")
    return "\n".join(lines)


def format_pipeline_report(report, *, capacity, units, has_observations):
    lines = ["DATA SET REPORT", "---------------"]
    for name, value in report.items():
        if name == "stage outcomes":
            continue
        if name == "capacity":
            name = (
                f"capacity estimate ({'kW' if value is not None else units})"
                if units
                else "capacity estimate"
            )
            if value is None:
                value = capacity
        if name == "length" and value is not None:
            value = f"{value:.2f} years"
        if name == "sampling" and value is not None:
            value = f"{value:g} minutes"
        lines.append(f"{name}: {'unavailable' if value is None else value}")
    outcomes = report["stage outcomes"]
    if details := format_outcomes(outcomes):
        lines.extend(("", details))
    if has_observations and any(
        outcome["status"] == "failed" for outcome in outcomes.values()
    ):
        lines.append(
            "Inspect plot_heatmap(matrix='raw') for the observations and gaps."
        )
    return "\n".join(lines)


def pipeline_error(outcomes):
    return (
        format_outcomes(
            {
                name: outcome
                for name, outcome in outcomes.items()
                if outcome["status"] == "failed"
            }
        )
        or "No error"
    )


def format_loss_report(report):
    lines = ["LOSS FACTOR REPORT", "------------------"]
    for name, value in report.items():
        precision = ".1f" if name.endswith("[kWh]") else ".3f"
        display = "unavailable" if value is None else format(value, precision)
        lines.append(f"{name}: {display}")
    return "\n".join(lines)
