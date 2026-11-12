"""Scale factors for supported electrical units."""

POWER_TO_W = {"W": 1, "kW": 1000}
ENERGY_TO_WH = {"Wh": 1, "kWh": 1000}


def unit_scale(source, target, scale_to_base):
    """Return the factor to express values in the target unit."""
    if source == target:
        return 1.0
    try:
        return scale_to_base[source] / scale_to_base[target]
    except KeyError:
        raise ValueError(f"Cannot convert from {source!r} to {target!r}.") from None
