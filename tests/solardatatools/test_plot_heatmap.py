import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from solardatatools import plot_heatmap


def test_plot_heatmap_standardizes_and_embeds_generic_data():
    times = pd.to_datetime(["2026-01-02 18:01", "2026-01-02 23:59", "2026-01-03 12:00"])
    frame = pd.DataFrame({"temperature": [2, 3, 4], "other": [-2, -3, -4]}, index=times)
    original = frame.copy()

    fig = plot_heatmap(frame, key="temperature", sampling_interval="6h")
    try:
        fig.canvas.draw()
        ax, colorbar = fig.axes
        image = ax.images[0]
        np.testing.assert_equal(
            image.get_array().filled(np.nan),
            [[np.nan, 3], [np.nan, np.nan], [np.nan, 4], [2, np.nan]],
        )
        assert image.get_cmap().name == "plasma"
        assert colorbar.get_ylabel() == ""
        first_day = mdates.date2num(pd.Timestamp("2026-01-02"))
        assert tuple(image.get_extent()) == (first_day - 0.5, first_day + 1.5, 24, 0)
        pd.testing.assert_frame_equal(frame, original)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("values", [[-2, 0, 5, None], [-2, -1, None, None]])
def test_plot_heatmap_centers_signed_data_on_zero(values):
    times = pd.date_range("2026-01-02", periods=4, freq="6h")
    data = pd.Series(values, index=times, dtype="Float64")

    fig = plot_heatmap(data, units="°C")
    try:
        ax, colorbar = fig.axes
        image = ax.images[0]
        limit = data.abs().max()
        assert image.get_cmap().name == "seismic"
        assert image.get_clim() == (-limit, limit)
        np.testing.assert_allclose(image.norm([-limit, 0, limit]), [0, 0.5, 1])
        assert colorbar.get_ylabel() == "°C"
    finally:
        plt.close(fig)
