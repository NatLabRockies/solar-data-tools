# -*- coding: utf-8 -*-
"""Matplotlib plots for timestamped signals and solar data analysis results.

Use :func:`plot_heatmap` for a generic time series with a datetime index. It
standardizes the timestamps and plots one column per day with missing samples
left blank. Use :func:`plot_2d` for an already embedded daily matrix and
:func:`plot_bundt_cake` for a 3D view of 365 daily profiles.

:class:`~solardatatools.data_handler.DataHandler` selects data, masks, dates,
and units for its plotting methods. The private rendering functions here
construct figures from those inputs. Diagnostics that interpret fitted model
components, such as capacity changes and time shifts, live with their algorithms.
"""

import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import CenteredNorm
from matplotlib.ticker import MaxNLocator

from solardatatools.matrix_embedding import make_2d
from solardatatools.time_axis_manipulation import standardize_time_axis


_HEATMAP_CMAP = "plasma"


def plot_heatmap(
    data,
    key=None,
    *,
    units=None,
    figsize=(12, 6),
    sampling_interval=None,
    snap_tolerance=None,
    verbose=True,
):
    """Plot a real-valued time series as a heatmap with one column per day.

    The timestamps are standardized to a regular grid and gaps appear blank.
    Nonnegative data use the default ``plasma`` colormap. If any plotted
    value is negative, ``seismic`` uses symmetric limits around zero. With
    ``units=None``, the colorbar label is blank.

    :param data: A pandas Series or DataFrame with a DatetimeIndex.
    :param key: Column to plot. Required for a DataFrame with multiple columns.
    :param units: Optional colorbar label. Values are plotted in their input units.
    :param figsize: Matplotlib figure size.
    :param sampling_interval: Fixed grid spacing, such as ``"15min"``.
        Omit to resolve the spacing from the timestamps.
    :param snap_tolerance: Maximum allowed timestamp displacement. Defaults
        to half the sampling interval.
    :param verbose: Report rejected timestamps and collisions.
    :return: Matplotlib Figure.

    .. code-block:: python

        from solardatatools import plot_heatmap

        fig = plot_heatmap(weather, key="temperature", units="°C")
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if not isinstance(data, pd.DataFrame) or not isinstance(
        data.index, pd.DatetimeIndex
    ):
        raise TypeError(
            "plot_heatmap requires a Series or DataFrame with a DatetimeIndex."
        )
    if key is None:
        if data.shape[1] != 1:
            raise ValueError("Specify key to select a single column to plot.")
        series = data.iloc[:, 0]
    else:
        series = data[key]
    if not isinstance(series, pd.Series) or pd.api.types.is_complex_dtype(series.dtype):
        raise ValueError("Select one real-valued column to plot.")

    values = series.to_numpy(dtype=float, na_value=np.nan)
    frame = pd.DataFrame(
        {"value": np.where(np.isfinite(values), values, np.nan)}, index=series.index
    )
    standardized, _ = standardize_time_axis(
        frame,
        sampling_interval=sampling_interval,
        snap_tolerance=snap_tolerance,
        verbose=verbose,
    )
    matrix, dates = make_2d(standardized, key="value", return_day_axis=True)
    if dates.empty:
        raise ValueError("No finite observations on the standardized grid to plot.")
    signed = np.nanmin(matrix) < 0
    cmap = "seismic" if signed else _HEATMAP_CMAP
    norm = CenteredNorm() if signed else None

    first_day, last_day = mdates.date2num(dates[[0, -1]])
    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=figsize)
        image = ax.imshow(
            matrix,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="none",
            origin="upper",
            extent=(first_day - 0.5, last_day + 0.5, 24, 0),
        )
        fig.colorbar(image, ax=ax, label=units)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True, min_n_ticks=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d", tz="UTC"))
        ax.set_xlabel("Date")
        ax.set_ylabel("Time of day")
        ax.set_yticks([0, 6, 12, 18, 24], ["00:00", "06:00", "12:00", "18:00", "24:00"])
    return fig


def plot_2d(
    D,
    figsize=(12, 6),
    units="kW",
    clear_days=None,
    dates=None,
    year_lines=False,
    ax=None,
    color="red",
):
    """
    A function for plotting the power heat map for solar power data

    :param D: PV power data arranged as a matrix, typically the output of
        `data_transforms.make_2d()`
    :param figsize: the size of the desired figure (passed to `matplotlib`)
    :param units: the units of the power data
    :param clear_days: a boolean array marking the location of clear days in
        the data set, typically the output of `clear_day_detection.find_clear_days()`
    :return: `matplotlib` figure
    """
    if D is not None:
        with sns.axes_style("white"):
            if ax is None:
                fig, ax = plt.subplots(nrows=1, figsize=figsize)
            else:
                fig = ax.get_figure()
            foo = ax.imshow(D, cmap=_HEATMAP_CMAP, interpolation="none", aspect="auto")
            ax.set_title("Measured power")
            plt.colorbar(foo, ax=ax, label=units)
            ax.set_xlabel("Day number")
            ax.set_yticks([])
            ax.set_ylabel("(sunset)        Time of day        (sunrise)")
            if clear_days is not None:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                use_day = clear_days
                days = np.arange(D.shape[1])
                y1 = np.ones_like(days[use_day]) * D.shape[0] * 0.999
                ax.scatter(days[use_day], y1, marker="|", color=color, s=2)
                ax.scatter(days[use_day], 0.995 * y1, marker="|", color=color, s=2)
                ax.scatter(days[use_day], 0.99 * y1, marker="|", color=color, s=2)
                ax.scatter(days[use_day], 0.985 * y1, marker="|", color=color, s=2)
                ax.scatter(days[use_day], 0.98 * y1, marker="|", color=color, s=2)
                ax.scatter(days[use_day], 0.975 * y1, marker="|", color=color, s=2)
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
        if dates is not None:
            if D.shape[1] >= 356 * 1.5:
                mask = np.logical_and(dates.month == 1, dates.day == 1)
                day_ticks = np.arange(D.shape[1])[mask]
                plt.xticks(day_ticks, dates[day_ticks].year)
                plt.xlabel("Year")
            else:
                mask = dates.day == 1
                day_ticks = np.arange(D.shape[1])[mask]
                plt.xticks(day_ticks, dates[day_ticks].month)
                plt.xlabel("Month")
            if year_lines:
                for d in day_ticks:
                    plt.axvline(d, ls="--", color="gray", linewidth=1)
        return fig
    else:
        return


def plot_bundt_cake(
    data,
    figsize=(12, 8),
    units="kW",
    inner_radius=1.0,
    slice_thickness=100,
    elev=45,
    azim=30,
    zoom=1.0,
    zscale=0.5,
    ax=None,
    cmap="coolwarm",
):
    """
    A function for plotting solar power data in a 3D “Bundt cake” style.
    Author: Mehmet Giray Ogut
    Date: June 11, 2025

    :param data: A 2D NumPy array of shape (365, N), where 365 is the number of days
    :param figsize: Size of the figure (width, height)
    :param units: Label for the z-axis
    :param inner_radius: Inner radius for the first slice
    :param slice_thickness: Total radial extent of all slices combined
    :param elev: Elevation angle for the 3D view
    :param azim: Azimuth angle for the 3D view
    :param zoom: Controls the box aspect ratio in x and y directions
    :param ax: Optional Matplotlib Axes object (3D projection). If None, a new figure is created
    :param cmap: Colormap used for the surface
    :return: Matplotlib Figure object
    """
    if data is None or data.shape[0] != 365:
        return

    with sns.axes_style("white"):
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.get_figure()

        num_days, slices_per_day = data.shape
        theta_days = np.linspace(0, 2 * np.pi, num_days, endpoint=True)
        theta_slices = np.linspace(0, 2 * np.pi, slices_per_day, endpoint=True)
        theta_grid, slice_grid = np.meshgrid(theta_days, theta_slices, indexing="xy")

        x = (inner_radius + slice_grid * (slice_thickness / slices_per_day)) * np.cos(
            theta_grid
        )
        y = (inner_radius + slice_grid * (slice_thickness / slices_per_day)) * np.sin(
            theta_grid
        )
        z = data.T

        ax.plot_surface(x, y, z, cmap=cmap, edgecolor="none")
        ax.grid(False)
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zlabel(units)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect([zoom, zoom, zscale])
        plt.tight_layout()
        return fig


def _plot_flagged_heatmap(data, flagged_days, title, dates, units, year_lines, figsize):
    fig = plot_2d(
        data,
        figsize=figsize,
        clear_days=flagged_days,
        dates=dates,
        year_lines=year_lines,
        units=units,
    )
    if title is not None:
        plt.title(f"Measured power, {title}")
    return fig


def _plot_daily_signals(
    xs,
    data,
    *,
    flags,
    model,
    figsize,
    color,
    alpha,
    label,
    mask_label,
    show_legend,
    marker,
):
    fig = plt.figure(figsize=figsize)
    kwargs = {}
    if color is not None:
        kwargs["color"] = color
    if alpha is not None:
        kwargs["alpha"] = alpha
    if marker is not None:
        kwargs["marker"] = marker
    if label is None:
        label = "measured power"
    plt.plot(xs, data, linewidth=1, **kwargs, label=label)
    if flags is not None:
        if mask_label is None:
            mask_label = "boolean mask"
        plt.plot(
            xs[flags],
            data[flags],
            ls="none",
            marker=".",
            color="red",
            label=mask_label,
        )
    if model is not None:
        plt.plot(xs, model, color="orange", linewidth=1, label="clear sky model")
    if show_legend:
        plt.legend()
    return fig


def _plot_density_signal(
    xs,
    density,
    *,
    flagged,
    suffix,
    seasonal_fit,
    lower_threshold,
    upper_threshold,
    figsize,
):
    fig = plt.figure(figsize=figsize)
    plt.plot(xs, density, linewidth=1)
    title = "Daily signal density"
    if suffix:
        plt.plot(xs[flagged], density[flagged], ls="none", marker=".", color="red")
        title += suffix
    if seasonal_fit is not None:
        plt.plot(xs, seasonal_fit, color="orange")
        plt.plot(
            xs,
            lower_threshold * seasonal_fit,
            color="green",
            linewidth=1,
            ls="--",
        )
        plt.plot(
            xs,
            upper_threshold * seasonal_fit,
            color="green",
            linewidth=1,
            ls="--",
        )
    plt.title(title)
    plt.gcf().autofmt_xdate()
    plt.ylabel("Fraction non-zero values")
    plt.xlabel("Date")
    return fig


def _plot_data_quality_scatter(
    density,
    linearity,
    labels,
    linearity_threshold,
    density_upper_threshold,
    density_lower_threshold,
    figsize,
):
    fig = plt.figure(figsize=figsize)
    for lb in set(labels):
        plt.scatter(
            density[labels == lb],
            linearity[labels == lb],
            marker=".",
            label=lb,
        )
    plt.xlabel("density score")
    plt.ylabel("linearity score")
    plt.axhline(
        linearity_threshold,
        linewidth=1,
        color="red",
        ls=":",
        label="decision boundary",
    )
    plt.axvline(density_upper_threshold, linewidth=1, color="red", ls=":")
    plt.axvline(density_lower_threshold, linewidth=1, color="red", ls=":")
    plt.legend()
    return fig


def _plot_daily_energy(xs, energy, *, electrical, units, flagged, suffix, figsize):
    fig = plt.figure(figsize=figsize)
    plt.plot(xs, energy, linewidth=1)
    title = "Daily energy production" if electrical else "Daily signal integral"
    if suffix:
        plt.plot(xs[flagged], energy[flagged], ls="none", marker=".", color="red")
        title += suffix
    plt.title(title)
    plt.gcf().autofmt_xdate()
    plt.xlabel("Date")
    label = "Energy" if electrical else "Integral"
    plt.ylabel(f"{label} ({units})" if units is not None else label)
    return fig


def _plot_circular_distribution(circ_data, num_bins, figsize, title):
    circ_hist = np.histogram(circ_data, bins=num_bins)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    start = (circ_hist[1][0] + circ_hist[1][1]) / 2
    end = (circ_hist[1][-1] + circ_hist[1][-2]) / 2
    theta = np.linspace(start, end, num_bins)
    radii = circ_hist[0]
    width = 2 * np.pi / num_bins
    bars = ax.bar(theta, radii, width=width, bottom=0.0, edgecolor="none")
    for r, bar in zip(radii, bars):
        bar.set_facecolor(cm.magma(r / np.max(circ_hist[0])))
        bar.set_alpha(0.75)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rorigin(-2.0)
    ax.set_rlabel_position(0)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    ax.set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    )
    ax.set_title(title)
    return fig


def _prepare_bundt_data(dilated_matrix, dates, *, aggregate, skip):
    if aggregate:
        # Drop Feb 29 entries
        leap_day_mask = np.logical_and(dates.month == 2, dates.day == 29)
        valid_days_mask = ~leap_day_mask
        valid_day_index = dates[valid_days_mask]
        valid_matrix = dilated_matrix[:, valid_days_mask]  # shape: (101, N')

        # Create 365-day vector for day-of-year (1–365, excluding Feb 29)
        doy = np.copy(valid_day_index.dayofyear.to_numpy())
        doy[doy > 59] -= 1  # adjust for removed Feb 29

        # Aggregate using nanmedian for robustness to missing data
        data_for_bundt = np.zeros((365, dilated_matrix.shape[0]))
        data_for_bundt[:] = np.nan  # initialize with NaNs

        for day in range(1, 366):
            same_day_mask = doy == day
            if np.any(same_day_mask):
                # shape: (101, n_years_for_day)
                day_data = valid_matrix[:, same_day_mask]
                data_for_bundt[day - 1] = np.nanmedian(day_data, axis=1)
    else:
        start_day = skip
        end_day = skip + 365
        data = dilated_matrix[:, start_day:end_day]
        if data.shape[1] < 365:
            # Pad with zeros if not enough days
            pad = 365 - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad)), constant_values=0)
        data_for_bundt = data.T  # shape: (365, 101)

    return data_for_bundt
