import marimo

__generated_with = "0.15.2"
app = marimo.App(width="columns", layout_file="layouts/SDT_demo.slides.json")


@app.cell(column=0)
def _(mo):
    static_polar_image = mo.image("assets/polar_transform_2.webp", width="27vw")
    return (static_polar_image,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    def topmd(prose: str, height="90vh", width="75vw") -> mo.Html:
        return mo.md(prose).style(height=height, width=width)

    return


@app.cell
def _():
    # scientific python imports
    import pandas as pd
    import numpy as np

    # plotting imports
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams["figure.dpi"] = 200

    # mapping import
    import folium

    # SDT imports
    from solardatatools import DataHandler

    return DataHandler, folium, np, pd, plt


@app.cell
def _():
    from solardatatools import plot_2d

    return (plot_2d,)


@app.cell
def _(pd, read_data):
    if read_data.value:
        df = pd.read_csv("2107_electrical_data.csv", index_col=0, parse_dates=[0])
        df = df[[_c for _c in df.columns if "ac_power" in _c]]
    else:
        df = pd.DataFrame()
    return (df,)


@app.cell
def _(mo):
    read_data = mo.ui.run_button(label="import data")
    return (read_data,)


@app.cell
def _(mo):
    run_sdt_button = mo.ui.run_button(label="run SDT pipeline")
    return (run_sdt_button,)


@app.cell
def _(mo):
    run_lossfa_button = mo.ui.run_button(label="run loss analysis")
    return (run_lossfa_button,)


@app.cell
def _(mo):
    slider_example = mo.ui.slider(0, 22, label="excitment level")
    return (slider_example,)


@app.cell
def _(DataHandler, mo):
    @mo.cache
    def run_sdt(df):
        dh = DataHandler(df)
        dh.fix_dst()
        dh.run_pipeline(verbose=True)
        return dh

    return (run_sdt,)


@app.function
# @mo.cache
def run_lossfa(dh):
    dh.run_loss_factor_analysis()
    return dh


@app.cell
def _(mo):
    heatmap_labels = mo.ui.dropdown(
        {
            "None": None,
            "Clear sky days": "clear",
            "Cloudy days": "cloudy",
            "System issues": "bad",
        },
        value="None",
        label="select day labels",
    )
    zoom_level = mo.ui.number(start=1, stop=5, label="zoom level")
    return heatmap_labels, zoom_level


@app.cell
def _(dh, heatmap_labels, mo, plt, zoom_level):
    try:
        heatmap = dh.plot_heatmap(
            "raw", flag=heatmap_labels.value, figsize=(8 * 1.25, 3 * 1.25)
        )
        _c = dh.num_days // 2 + 50
        if zoom_level.value == 1:
            pass
        elif zoom_level.value == 2:
            plt.xlim(_c - 365 * 1, _c + 356 * 1)
        elif zoom_level.value == 3:
            plt.xlim(_c - 365 * 0.5, _c + 356 * 0.5)
        elif zoom_level.value == 4:
            plt.xlim(_c - 365 * 0.25, _c + 356 * 0.25)
        elif zoom_level.value == 5:
            plt.xlim(_c - 365 * 0.125, _c + 356 * 0.125)
    except:
        heatmap = mo.md("Run pipeline to view plot")
    return (heatmap,)


@app.cell
def _():
    # dh.plot_bundt()
    # _ax = plt.gca()
    # for _ix in range(72):
    #     _ax.view_init(elev=45, azim=45+5*_ix, roll=0)
    #     plt.gcf().savefig(f'bundt-{_ix:02}.png')
    return


@app.cell
def _(dh):
    try:
        cap_change_plot = dh.plot_capacity_change_analysis(figsize=(8, 3.5))
        cap_change_plot.axes[0].set_ylim(19, 31)
    except:
        cap_change_plot = None
    return (cap_change_plot,)


@app.cell
def _(dh, plt):
    try:
        inverter_clip_plot = dh.plot_clipping(figsize=(8, 4))
        plt.tight_layout()
        # inverter_clip_plot.axes[0].set_ylim(19, 31)
    except:
        inverter_clip_plot = None
    return (inverter_clip_plot,)


@app.cell
def _(dh, plt):
    try:
        inverter_clip_plot2 = dh.plot_daily_max_cdf_and_pdf(
            figsize=(6 * 1.05, 4 * 1.05)
        )
        plt.tight_layout()
        # inverter_clip_plot.axes[0].set_ylim(19, 31)
    except:
        inverter_clip_plot2 = None
    return (inverter_clip_plot2,)


@app.cell
def _(folium):
    lat, lon = 38.996306, -122.134111
    m = folium.Map(location=[lat, lon], zoom_start=17, tiles="OpenStreetMap")
    folium.TileLayer("Esri.WorldImagery").add_to(m)
    folium.Marker(
        location=[lat, lon], popup="<b>Site 2107 “Farm Solar Array (CA)”</b>"
    ).add_to(m)
    folium.LayerControl().add_to(m);
    return lat, lon, m


@app.cell
def _(dh):
    try:
        circ_dist = dh.plot_circ_dist(flag="clear")
    except:
        circ_dist = None
    return (circ_dist,)


@app.cell
def _(dh_lfa):
    try:
        lfa_plot = dh_lfa.loss_analysis.plot_decomposition(figsize=(12, 10))
    except:
        lfa_plot = None
    return (lfa_plot,)


@app.cell
def _(dh_lfa):
    try:
        loss_waterfall = dh_lfa.loss_analysis.plot_waterfall()
    except:
        loss_waterfall = None
    return (loss_waterfall,)


@app.cell
def _(dh_lfa):
    try:
        loss_pie = dh_lfa.loss_analysis.plot_pie()
    except:
        loss_pie = None
    return (loss_pie,)


@app.cell
def _(lat, lon, mo, reset_lat_lon):
    reset_lat_lon.value
    lat_sldr = mo.ui.slider(
        start=-90, stop=90, value=lat, label="latitude", show_value=True, debounce=True
    )
    lon_sldr = mo.ui.slider(
        start=-180,
        stop=180,
        value=lon,
        label="longitude",
        show_value=True,
        debounce=True,
    )
    lat_lon_sldrs = mo.md("""
        {lat}

        {lon}
    """).batch(
        lat=lat_sldr,
        lon=lon_sldr,
    )
    return (lat_lon_sldrs,)


@app.cell
def _(mo):
    reset_lat_lon = mo.ui.run_button(label="reset values")
    return (reset_lat_lon,)


@app.cell
def _(dh, lat_lon_sldrs, plt):
    try:
        dh.plot_polar_transform(
            lat=lat_lon_sldrs.value["lat"], lon=lat_lon_sldrs.value["lon"], tz_offset=-8
        )
        polar_plot = plt.gcf()
    except:
        polar_plot = None
    return (polar_plot,)


@app.cell
def _(dh):
    try:
        dh.setup_location_and_orientation_estimation(-8)
        tilt, az = dh.estimate_orientation()
    except:
        tilt, az = None, None
    return az, tilt


@app.cell
def _(dh, mo):
    try:
        start_day_slider = mo.ui.slider(
            start=0,
            stop=dh.num_days,
            value=100,
            label="start day",
            full_width=True,
            show_value=True,
        )
        num_day_slider = mo.ui.slider(
            start=1,
            stop=dh.num_days,
            value=5,
            label="number of days",
            full_width=True,
            show_value=True,
        )
    except:
        start_day_slider = mo.ui.slider(0, 1)
        num_day_slider = mo.ui.slider(0, 1)
    return num_day_slider, start_day_slider


@app.cell
def _(mo):
    show_clipped_times = mo.ui.switch(label="show clipped times")
    return (show_clipped_times,)


@app.cell
def _(dh, plt):
    try:
        daily_energy_plot = dh.plot_daily_energy()
        plt.ylabel("Energy (kWh)")
        daily_energy_sig = dh.daily_signals.energy
    except:
        daily_energy_plot = None
        daily_energy_sig = None
    return daily_energy_plot, daily_energy_sig


@app.cell
def _(dh, num_day_slider, plt, show_clipped_times, start_day_slider):
    try:
        if not show_clipped_times.value:
            ts_plot = dh.plot_daily_signals(
                start_day=start_day_slider.value,
                num_days=num_day_slider.value,
                figsize=(12, 4),
                show_clear_model=False,
            )
        else:
            ts_plot = dh.plot_daily_signals(
                start_day=start_day_slider.value,
                num_days=num_day_slider.value,
                boolean_mask=dh.boolean_masks.clipped_times,
                mask_label="clipped",
                show_legend=True,
                figsize=(12, 4),
                show_clear_model=False,
            )
            plt.legend(loc=4)
        plt.xticks(rotation=45)
    except:
        ts_plot = None
    return (ts_plot,)


@app.cell
def _(mo):
    elev_sldr = mo.ui.slider(
        start=-180, stop=180, value=45, label="elevation", show_value=True
    )
    azim_sldr = mo.ui.slider(
        start=0, stop=360, value=45, label="azimuth", show_value=True
    )
    roll_sldr = mo.ui.slider(
        start=-180, stop=180, value=0, label="roll", show_value=True
    )
    return azim_sldr, elev_sldr, roll_sldr


@app.cell
def _(azim_sldr, dh, elev_sldr, plt, roll_sldr):
    try:
        bundt_plot = dh.plot_bundt()
        _ax = plt.gca()
        _ax.view_init(elev=elev_sldr.value, azim=azim_sldr.value, roll=roll_sldr.value)
    except:
        bundt_plot = None
    return (bundt_plot,)


@app.cell
def _(mo):
    fit_clearsky_button = mo.ui.run_button(label="fit clearsky model")
    return (fit_clearsky_button,)


@app.function
def fit_clearsky(dh):
    dh.fit_statistical_clear_sky_model(verbose=False)
    return dh


@app.cell
def _(dh, mo):
    try:
        start_day_slider2 = mo.ui.slider(
            start=0,
            stop=dh.num_days,
            value=700,
            label="start day",
            full_width=True,
            show_value=True,
        )
        num_day_slider2 = mo.ui.slider(
            start=1,
            stop=dh.num_days,
            value=5,
            label="number of days",
            full_width=True,
            show_value=True,
        )
    except:
        start_day_slider2 = mo.ui.slider(0, 1)
        num_day_slider2 = mo.ui.slider(0, 1)
    return num_day_slider2, start_day_slider2


@app.cell
def _(dh_cs):
    try:
        dh_cs.detect_clear_sky()
    except:
        pass
    return


@app.cell
def _(mo):
    show_clear_times = mo.ui.switch(label="show clear sky times")
    return (show_clear_times,)


@app.cell
def _(dh_cs, num_day_slider2, plt, show_clear_times, start_day_slider2):
    try:
        if show_clear_times.value:
            ts_plot2 = dh_cs.plot_daily_signals(
                start_day=start_day_slider2.value,
                num_days=num_day_slider2.value,
                figsize=(12, 4),
                show_clear_model=True,
                boolean_mask=dh_cs.boolean_masks.clear_times,
                mask_label="clear sky periods",
                show_legend=True,
            )
        else:
            ts_plot2 = dh_cs.plot_daily_signals(
                start_day=start_day_slider2.value,
                num_days=num_day_slider2.value,
                figsize=(12, 4),
                show_clear_model=True,
                show_legend=True,
            )
        plt.legend(loc=4)
        plt.xticks(rotation=45)
    except:
        ts_plot2 = None
    return (ts_plot2,)


@app.cell
def _(dh, dh_cs, mo, plot_2d, plt):
    try:
        cs_heatmap = plot_2d(
            dh_cs.quantile_object.quantiles_original[0.9].reshape(
                (-1, dh.num_days), order="F"
            ),
            figsize=(8 * 1.25, 3 * 1.25),
        )
        plt.title("Estimated clear sky power")
    except:
        cs_heatmap = None
    try:
        heatmap2 = dh.plot_heatmap("raw", figsize=(8 * 1.25, 3 * 1.25))
    except:
        heatmap2 = mo.md("Run pipeline to view plot")
    return cs_heatmap, heatmap2


@app.cell
def _(mo):
    elev_sldr2 = mo.ui.slider(
        start=-180, stop=180, value=45, label="elevation", show_value=True
    )
    azim_sldr2 = mo.ui.slider(
        start=0, stop=360, value=45, label="azimuth", show_value=True
    )
    roll_sldr2 = mo.ui.slider(
        start=-180, stop=180, value=0, label="roll", show_value=True
    )
    return azim_sldr2, elev_sldr2, roll_sldr2


@app.cell
def _(azim_sldr2, dh, elev_sldr2, plt, roll_sldr2):
    try:
        bundt_plot2 = dh.plot_bundt()
        _ax = plt.gca()
        _ax.view_init(
            elev=elev_sldr2.value, azim=azim_sldr2.value, roll=roll_sldr2.value
        )
    except:
        bundt_plot2 = None
    return (bundt_plot2,)


@app.cell
def _(azim_sldr2, dh_cs, elev_sldr2, plt, roll_sldr2):
    try:
        bundt_q = dh_cs.quantile_object.plot_quantile_bundt(quantile=0.9)
        _ax = plt.gca()
        _ax.view_init(
            elev=elev_sldr2.value, azim=azim_sldr2.value, roll=roll_sldr2.value
        )
    except:
        bundt_q = None
    return (bundt_q,)


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(
        f"""
    <h1 style="text-align: center;line-height:1;">Solar Data Tools Live Tutorial</h1>
    <h3 style="text-align: center;line-height:1;">September 25, 2025</h3>
    <h3 style="text-align: center;line-height:1;">Bennet Meyers, PhD</h3>
    <br>
    <p style="text-align: center;line-height:1;">National Renewable Energy Laboratory</p>
    <p style="text-align: center;line-height:1;">Golden, CO</p>
    <br>
    {mo.image(src="assets/SDT_v1_secondary_blue_text.png", width="20vw").center()}

    {
            mo.hstack(
                [
                    mo.image(src="assets/nrel-blue-logo.png", width="8vw"),
                    mo.image(src="assets/SLAC_primary_red.png", width="10vw"),
                    mo.image(src="assets/SUSig_StnfrdOnly_red2.png", width="8vw"),
                ],
                justify="center",
                gap=3,
            )
        }
    """
    )
    return


@app.cell
def _(mo, slider_example):
    _title = mo.md("## 👋 Welcome! ")
    if slider_example.value < 1:
        _lc = mo.md(
            r"""
        ### Agenda
        **Now:**

        - introduction to Solar Data Tools and overview of basic concepts
        - code demonstration of basic usage and API
            - running the initial pipeline
            - data summary and basic plots
            - inspecting the results of the pipeline algorithms as plots
            - loss factor and clear sky analysis
        - we will be executing code live here in the presentation, thanks to `marimo` (https://marimo.io)

        **Later:**

        - large data pipelines with SDT from PV Fleets Team
        - open-source community building
        """
        )
    else:
        _lc = mo.md(
            r"""
        ### Agenda
        **Now:**

        - introduction to Solar Data Tools and overview of basic concepts
        - code demonstration of basic usage and API
            - running the initial pipeline
            - data summary and basic plots
            - inspecting the results of the pipeline algorithms as plots
            - loss factor and clear sky analysis
        - we will be executing code ✨ live ✨ here in the presentation, thanks to `marimo` (https://marimo.io)

        **Later:**

        - large data pipelines with SDT from PV Fleets Team
        - open-source community building
        """
        )
    _rc = mo.vstack([slider_example, mo.md("##" + "✨" * slider_example.value)])
    _cols = mo.hstack(
        [_lc, _rc], justify="space-between", align="center", gap=2, widths=[2, 1]
    )
    mo.vstack([_title, _cols], align="start", justify="start", gap="5")
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.md("# Background and overview"),
            mo.image(
                src="assets/SDT_v1_secondary_blue_text.png", width="10vw"
            ).center(),
        ],
        widths=[1, 1],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    _title = mo.md("## SDT: basic idea")
    _lc = mo.md(
        r"""
    - open-source Python library (BSD 2.0) for analyzing PV power (and irradiance) time-series data
    - analyze **unlabeled** power data—no system model, meteorological data, or performance index
    - single-channel analysis—the only requirement is a power signal
    - use **signal processing** and **physics-informed machine learning** techniques to model data
    - intended to be run “out of the box,” little to no babysitting of algorithm parameters
    - the process:
        - load data as `pandas` data frame, timestamps in index
        - run basic onboarding pipeline
        - perform further analysis, calculation, or plotting
    """
    )
    _rc = mo.image("assets/bundt.gif", width="50vw")
    _cols = mo.hstack([_lc, _rc], justify="space-between", gap=5, widths=[1, 1])
    mo.vstack([_title, _cols], align="start", justify="start", gap="5")
    return


@app.cell
def _(mo):
    _title = mo.md("## A brief history...")
    _lc = mo.md(
        r"""
    - began development in February 2019
    - started as part of PhD dissertation on statistical signal processing
    - (see “Signal Decomposition Using Masked Proximal Operators,” `doi: 10.1561/2000000122`)
    - inspired by practical experience of 8 years in solar industry (SunPower, Performance R&D)
    - desire to streamline analysis of PV data, for quicker insights, less data munging, and easier large-scale pipelines
    - now up to 20 unique code contributors
    - more on our open-source developments and community from Sara in a bit
    """
    )
    _rc = mo.image("assets/star-history-2025922.png", width="40vw")
    _cols = mo.hstack([_lc, _rc], justify="space-between", gap=5, widths=[1, 1])
    mo.vstack([_title, _cols], align="start", justify="start", gap="5")
    return


@app.cell
def _(mo):
    _title = mo.md("## What SDT gives you")
    _lc = mo.md(
        r"""
    - in about **15 seconds** (per data stream): data set summary, outage/data quality analysis, capacity change detection, basic plots
    - in **1–2 minutes**: automatic loss factor disaggregation, soiling and degradation loss estimation
    - in **5–10 minutes**: statistically derived model of clear sky response, sub-daily clear period labeling
    - (timing with open-source solvers, much faster with commercial solvers)
    - all **automatically** with only a power signal and no user input
    - one or two lines of code

    """
    )
    _rc = mo.image("assets/lfa_plot.png", width="70vw")
    _cols = mo.hstack([_lc, _rc], justify="space-between", gap=5, widths=[1, 1])
    mo.vstack([_title, _cols], align="start", justify="start", gap="5")
    return


@app.cell
def _(mo):
    _title = mo.md("## What can solar data tools do for you?")
    # elevator pitch
    return


@app.cell
def _(mo):
    _title = mo.md("## Standard data onboarding pipeline")
    _lc = mo.md(
        r"""
    ### Pipeline steps
    - **preprocessing**
        - time stamp cleaning and time axis standardization
        - matrix embedding (*i.e.*, heatmap view)
    - **cleaning**
        - fill missing data
        - time shift detection and correction (optional)
        - large timezone error detection
    - **filtering/labeling**
        - data quality / missing data
        - clear / cloudy day labeling
        - inverter clipping detection / labeling
        - system capacity change detection
    """
    ).style(width="40vw")
    _rc = mo.md(
        r"""
    ### Discussion
    - all subroutines can be used independently, programatically
    - object oriented components
    - standard pipeline automates the sequestion execution of many subroutines (some of which depend on each other)
    - filtering/labeling steps all involve ML/SP methods
    """
    )
    _cols = mo.hstack([_lc, _rc], justify="center", gap=2, widths=[1, 1])
    mo.vstack([_title, _cols], align="start", justify="start", gap="5")
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.md("# Running the pipeline"),
            mo.image(
                src="assets/SDT_v1_secondary_blue_text.png", width="10vw"
            ).center(),
        ],
        widths=[1, 1],
        align="center",
    )
    return


@app.cell
def _(m, mo):
    _title = mo.md("## Data example")
    _lc = mo.md(
        r"""
    - [Solar Data Prize, Site 2107](https://openei.org/wiki/PVDAQ/Sites/Farm_Solar_Array) “Farm Solar Array”
    - Arbuckle, CA
    - mono-Si
    - ground-mount, fixed-tilt (az: 180°, tilt 25°)
    """
    ).style(width="40vw")
    _rc = m
    _cols = mo.hstack([_lc, _rc], justify="space-between", gap=5)
    mo.vstack([_title, _cols], align="start", justify="start", gap="5")
    return


@app.cell(hide_code=True)
def _(df, mo, read_data):
    mo.vstack(
        [
            mo.md("""
        ## Load data
        Solar data tools expects data as a Pandas data frame, with timestamps in the index, marking the local time

        ~~~
        df = pd.read_csv("2107_electrical_data.csv", index_col=0, parse_dates=[0])
        ~~~
        """),
            read_data,
            df,
        ]
    )
    return


@app.cell(hide_code=True)
def run_sdt_pipeline(mo, run_sdt_button):
    _title = mo.md("## Run SDT pipeline")
    _lc = mo.vstack(
        [
            mo.md("""
        ```python
        dh = DataHandler(df)
        dh.fix_dst()
        dh.run_pipeline()
        ```
        """),
            run_sdt_button,
        ]
    )
    _rc = mo.md("""
    - instatiate `DataHandler` class on data frame
    - optional `.fix_dst()` method automatically removes daylight saving times
    - can pass optional column name through `power_col` kwarg otherwise `.run_pipeline` assumes the first column
    """)
    _cols = mo.hstack([_lc, _rc], justify="space-between", gap=5, widths=[1, 1])
    mo.vstack([_title, _cols], align="start", justify="start", gap="5")
    return


@app.cell(hide_code=True)
def _(df, mo, run_sdt, run_sdt_button):
    if not run_sdt_button.value:
        mo.output.replace(
            mo.vstack(
                [
                    mo.md("## Run SDT pipeline"),
                    mo.md("""
                ```python
                dh = DataHandler(df)
                dh.fix_dst()
                dh.run_pipeline()
                ```
                """),
                    mo.md("#### Click button on previous slide to begin! 🔙"),
                ]
            )
        )
        dh = None
    else:
        mo.output.replace(mo.md("## Run SDT pipeline"))
        with mo.status.spinner():
            with mo.redirect_stdout():
                with mo.redirect_stderr():
                    dh = run_sdt(df)
        mo.output.append(mo.md("#### Done! 🎉"))
    return (dh,)


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.md("# Data summary and overview with SDT"),
            mo.image(
                src="assets/SDT_v1_secondary_blue_text.png", width="10vw"
            ).center(),
        ],
        widths=[1, 1],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(dh, mo):
    _text = """
    ## Get human- or machine-readable summary right away

    ```
    dh.report()
    ```
    prints the following to `stout`:
    """
    mo.output.replace(mo.md(_text))
    with mo.redirect_stdout():
        try:
            dh.report()
        except:
            pass
    _text2 = """
    ```
    dh.report(return_values=True)
    ```
    returns a dictionary of values:
    """
    mo.output.append(mo.md(_text2))
    try:
        mo.output.append(dh.report(return_values=True))
    except:
        pass
    return


@app.cell
def _(mo, num_day_slider, show_clipped_times, start_day_slider, ts_plot):
    # mo.output.replace(heatmap_labels)
    # mo.output.append(zoom_level)
    # mo.output.append(heatmap)
    if not show_clipped_times.value:
        _s = f"dh.plot_daily_signals(start_day={start_day_slider.value}, num_days={num_day_slider.value})"
    else:
        _s = f"dh.plot_daily_signals(start_day={start_day_slider.value}, num_days={num_day_slider.value}, boolean_mask=dh.boolean_masks.clipped_times, mask_label='clipped', show_legend=True)"
    mo.vstack(
        [
            mo.md("## View time series plot"),
            start_day_slider,
            num_day_slider,
            show_clipped_times,
            mo.as_html(mo.md("\n".join(["```", _s, "```"]))).style(width="70vw"),
            mo.as_html(ts_plot).style(width="55vw").center(),
        ]
    )
    return


@app.cell
def _(heatmap, heatmap_labels, mo, zoom_level):
    # mo.output.replace(heatmap_labels)
    # mo.output.append(zoom_level)
    # mo.output.append(heatmap)
    _s = f"dh.plot_heatmap('raw', flag={heatmap_labels.value})"
    mo.vstack(
        [
            mo.md("## View data as a heatmap, inspect labeled days"),
            heatmap_labels,
            zoom_level,
            mo.md("\n".join(["```", _s, "```"])),
            mo.as_html(heatmap).style(width="50vw").center(),
        ]
    )
    return


@app.cell
def _(azim_sldr, bundt_plot, elev_sldr, mo, roll_sldr):
    _s = f"""
    dh.plot_bundt()
    ax = plt.gca()
    ax.view_init(elev={elev_sldr.value}, azim={azim_sldr.value}, roll={roll_sldr.value})
    """
    _l = mo.md(
        """
    - emphasizes seasonal structure in data
    - removes night time data, standardizes day length (inner/outer diameter)
    - height/color show power production
    - see “Time dilated Bundt cake analysis of PV output” (2024 PVSC)
    """
    )
    mo.vstack(
        [
            mo.md("## Bundt plot 🍰"),
            elev_sldr,
            azim_sldr,
            roll_sldr,
            mo.md("\n".join(["```python", _s, "```"])),
            mo.hstack(
                [mo.as_html(bundt_plot).style(width="30vw").center(), _l],
                widths=[1, 1],
                align="center",
                gap=5,
            ),
        ],
    )
    return


@app.cell
def _(lat_lon_sldrs, mo, polar_plot, reset_lat_lon, static_polar_image):
    _title = mo.md("## Polar transform view")
    _top = mo.md(
        f"""
    ```
    dh.plot_polar_transform(lat={lat_lon_sldrs.value["lat"]}, lon={lat_lon_sldrs.value["lon"]}, tz_offset=-8)
    ```

    - automatically map measured power/irradiance to solar sky position
    - provides visual of site obstructions
    - requires (approximate) lat/lon, can be read from file or estimate from measured power with just a GMT offset

    ```
    dh.setup_location_and_orientation_estimation(-8)
    lat_est, lon_est = dh.estimate_latitude(), dh.estimate_longitude()
    ```
    """
    )
    _lc = mo.vstack(
        [
            mo.md("#### inverter power (live)").center(),
            mo.as_html(polar_plot).style(width="26vw"),
        ]
    )
    _rc = mo.vstack(
        [mo.md("#### pyranometer irradiance (static)").center(), static_polar_image]
    )
    _cols = mo.hstack([_lc, _rc], justify="space-between", gap=5, widths=[1, 1])
    mo.vstack(
        [_title, mo.hstack([lat_lon_sldrs, reset_lat_lon]), _top, _cols],
        align="start",
        justify="start",
        gap=0.25,
    )
    return


@app.cell
def _(az, circ_dist, daily_energy_plot, daily_energy_sig, mo, np, tilt):
    _title = mo.md("## Other data summary operations")
    if tilt is None:
        _tilt = np.nan
    else:
        _tilt = tilt
    if az is None:
        _az = np.nan
    else:
        _az = az
    _lc = mo.md(
        f"""
    - for uniformly orientied fix tilt systems, estimate array tilt and azimuth (south is zero, west is `+`):

    ```
    dh.setup_location_and_orientation_estimation(gmt_offset)
    dh.estimate_orientation()
    ```

    |          | tilt     | azimuth |
    | -------- | -------- | ------- |
    | estimate |{_tilt:.1f}| {_az:.1f}|
    | actual   | 25.0     | 0.0     |

    - daily energy signal as `numpy.array` object

    ```
    dh.daily_signals.energy
    ```

    {daily_energy_sig}

    - complete list of plots at https://solar-data-tools.readthedocs.io/en/stable/index_user_guide.html#plotting-some-pipeline-results
    """
    )
    _mc = mo.md(
        f"""
    - plot circular histograms of data labels, *e.g.*, inspect seasonal pattern of clear days

    ```
    dh.plot_circ_dist(flag='clear')
    ```
    {mo.as_html(circ_dist).style(width="17vw").center()}

    """
    )
    _rc = mo.md(
        f"""
    - plot daily energy:

    ```
    dh.plot_daily_energy()
    ```

    {mo.as_html(daily_energy_plot).style(width="22vw").center()}

    """
    )
    mo.vstack([_title, mo.hstack([_lc, _mc, _rc], widths="equal", gap=5)], gap=1)
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.md("# Inspection of pipeline ML algorithms"),
            mo.image(
                src="assets/SDT_v1_secondary_blue_text.png", width="10vw"
            ).center(),
        ],
        widths=[1, 1],
        align="center",
    )
    return


@app.cell
def _(cap_change_plot, mo):
    _t = """
    ## Capacity change analysis

    ```
    dh.plot_capacity_change_analysis()
    ```

    - detects abrupt shifts in apparent system capacity
    - generalized changepoint detection problem with adaptive seasonal term
    """
    mo.vstack([mo.md(_t), mo.as_html(cap_change_plot).style(width="55vw").center()])
    return


@app.cell
def _(inverter_clip_plot, inverter_clip_plot2, mo):
    _t = """
    ## Inverter clipping analysis

    ```
    dh.plot_clipping()
    dh.plot_daily_max_cdf_and_pdf()
    ```

    - detects days with significant clipping, more than 10% of the daily energy produced by clipped
    - handles multiple clipping set points in the data set
    - machine-readable labels to isolate clipped days (and separate by clipping level when multiple)
    """
    mo.vstack(
        [
            mo.md(_t),
            mo.hstack(
                [
                    mo.as_html(inverter_clip_plot).style(width="37vw").center(),
                    mo.as_html(inverter_clip_plot2).style(width="25vw").center(),
                ],
                gap=1,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    # PVDAQ site 3244
    _t = """
    ## A more complex example
    """
    mo.vstack(
        [
            mo.md(_t),
            mo.image(src="assets/multiclip_heatmap.png", width="40vw").center(),
            mo.hstack(
                [
                    mo.image(
                        src="assets/multiclip_metrics.png", height="17vw", width="45vw"
                    ).center(),
                    mo.image(src="assets/multiclip_cdf.png", height="15vw").center(),
                ],
                gap=0.25,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.md("# Further analysis"),
            mo.image(
                src="assets/SDT_v1_secondary_blue_text.png", width="10vw"
            ).center(),
        ],
        widths=[1, 1],
        align="center",
    )
    return


@app.cell
def _(mo, run_lossfa_button):
    _title = mo.md("## Run loss factor analysis")
    _lc = mo.vstack(
        [
            mo.md("""
        ```python
        dh.run_loss_factor_analysis()
        ```
        """),
            run_lossfa_button,
        ]
    )
    _rc = mo.md("""
    - decomposes measured energy into loss components and a seasonal (clear sky) baseline
    - losses: weather, capacity shifts, soiling, and long-term degradation
    - no additional data or configuration required
    - easily automated/included in larger pipeline
    """)
    _cols = mo.hstack([_lc, _rc], justify="space-between", gap=5, widths=[1, 1])
    mo.vstack([_title, _cols], align="start", justify="start", gap="5")
    return


@app.cell
def _(dh, mo, run_lossfa_button):
    # with mo.redirect_stdout():
    #     with mo.capture_stderr() as buffer:
    #         with mo.status.spinner():
    #             dh.run_loss_factor_analysis()

    if not run_lossfa_button.value:
        mo.output.replace(
            mo.vstack(
                [
                    mo.md("## Run loss factor analysis"),
                    mo.md("""
                ```python
                dh.run_loss_factor_analysis()
                ```
                """),
                    mo.md("#### Click button on previous slide to begin! 🔙"),
                ]
            )
        )
    else:
        mo.output.replace(mo.md("## Run loss factor analysis"))
        with mo.redirect_stdout():
            with mo.capture_stderr() as _buffer:
                with mo.status.spinner():
                    dh_lfa = run_lossfa(dh)
        mo.output.append(mo.md("#### Done! 🎉"))
    return (dh_lfa,)


@app.cell
def _(lfa_plot, mo):
    _t = """
    ## Loss component decomposition

    ```
    dh.loss_analysis.plot_decomposition()
    ```
    """
    mo.vstack(
        [mo.md(_t), mo.as_html(lfa_plot).style(height="15", width="40vw").center()]
    )
    return


@app.cell
def _(loss_pie, loss_waterfall, mo):
    _t = """
    ## View losses as waterfall or pie chart

    ```
    dh.loss_analysis.plot_decomposition()
    ```
    """
    mo.vstack(
        [
            mo.md(_t),
            mo.hstack(
                [
                    mo.as_html(loss_waterfall).style(width="40vw"),
                    mo.as_html(loss_pie).style(width="30vw"),
                ],
                align="center",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    _t = mo.md(
        """
    solve with clarabel: 8:21
    solve with mosek: 0:43
    """
    )
    return


@app.cell
def _(fit_clearsky_button, mo):
    _title = mo.md("## Fit clear sky model")
    _lc = mo.vstack(
        [
            mo.md("""
        ```python
        # dh.fit_statistical_clear_sky_model() # ~8 minutes
        dh.fit_statistical_clear_sky_model(solver='MOSEK') # ~45 seconds
        ```
        """),
            fit_clearsky_button,
        ]
    )
    _rc = mo.md("""
    - fits smooth, multi-periodic quantile model to the power data
    - see (again) “Time dilated Bundt cake analysis of PV output” (2024 PVSC)
    - we take the daily and seasonally adjusted 90th percentile as the clear sky response of the system
    - data driven baseline of system performance, can be used to construct a (kind of) performance index
    - also used directly to label clear sky periods in the data
    """)
    _cols = mo.hstack(
        [_lc, _rc], justify="space-between", gap=5, widths=[1, 1], align="center"
    )
    mo.vstack([_title, _cols], align="start", justify="start", gap="5")
    return


@app.cell
def _(dh, fit_clearsky_button, mo):
    if not fit_clearsky_button.value:
        mo.output.replace(
            mo.vstack(
                [
                    mo.md("## Fit clear sky model"),
                    mo.md("""
                ```python
                # dh.fit_statistical_clear_sky_model() # ~8 minutes
                dh.fit_statistical_clear_sky_model(solver='MOSEK') # ~45 seconds
                ```
                """),
                    mo.md("#### Click button on previous slide to begin! 🔙"),
                ]
            )
        )
    else:
        mo.output.replace(mo.md("## Fit clear sky model"))
        with mo.redirect_stdout():
            with mo.capture_stderr() as _buffer:
                with mo.status.spinner():
                    dh_cs = fit_clearsky(dh)
        mo.output.append(mo.md("#### Done! 🎉"))
    return (dh_cs,)


@app.cell
def _(mo, num_day_slider2, show_clear_times, start_day_slider2, ts_plot2):
    if show_clear_times.value:
        _s = f"dh.plot_daily_signals(start_day={start_day_slider2.value}, num_days={num_day_slider2.value}, show_clear_model=True, boolean_mask=dh.boolean_masks.clear_times, mask_label='clear sky periods', show_legend=True)"
    else:
        _s = f"dh.plot_daily_signals(start_day={start_day_slider2.value}, num_days={num_day_slider2.value}, show_clear_model=True, show_legend=True)"
    mo.vstack(
        [
            mo.md("## View clear sky signal on time series plot"),
            start_day_slider2,
            num_day_slider2,
            show_clear_times,
            mo.as_html(mo.md("\n".join(["```", _s, "```"]))).style(width="70vw"),
            mo.as_html(ts_plot2).style(width="55vw").center(),
        ]
    )
    return


@app.cell
def _(cs_heatmap, heatmap2, mo):
    _t = mo.md("## Compare heatmaps")
    _b = mo.hstack([heatmap2, cs_heatmap], widths=[1, 1], gap=5)
    mo.vstack([_t, _b], gap=5)
    return


@app.cell
def _(azim_sldr2, bundt_plot2, bundt_q, elev_sldr2, mo, roll_sldr2):
    _t = mo.md("## Compare Bundt plots")
    _l = mo.md(
        """
    - emphasizes seasonal structure in data
    - removes night time data, standardizes day length (inner/outer diameter)
    - height/color show power production
    - see “Time dilated Bundt cake analysis of PV output” (2024 PVSC)
    """
    )
    mo.vstack(
        [
            _t,
            elev_sldr2,
            azim_sldr2,
            roll_sldr2,
            mo.hstack(
                [
                    mo.as_html(bundt_plot2).style(width="30vw").center(),
                    mo.as_html(bundt_q).style(width="30vw").center(),
                ],
                widths=[1, 1],
                align="center",
                gap=5,
            ),
        ],
    )
    return


@app.cell
def _(mo):
    mo.hstack(
        [
            mo.md("# Wrapping it up..."),
            mo.image(
                src="assets/SDT_v1_secondary_blue_text.png", width="10vw"
            ).center(),
        ],
        widths=[1, 1],
        align="center",
    )
    return


@app.cell
def _(mo):
    _title = mo.md("## Recent work and accomplishments")
    _lc = mo.md(
        r"""
    - recently released `v2.0.1` (https://github.com/NREL/solar-data-tools/releases)
    - `v1` (September 2023) finalized the behavior of the main onboarding pipeline and added support for open-source solvers
    - `v2` (July 2025) finalized the loss factor analysis and clear sky estimation/detection, with support for open-source solvers
    - along the way, many improvements to stability and performance of ML/SP subroutines
    - development of CI/CD pipeline, improved test coverage, and documentation
    - still developing new, cutting-edge research such as **fleet-scale partial outage detection** (currently an open PR, won't present today, but see recent PVSC paper)
    """
    )
    _rc = mo.image("assets/fleet_outage_collage.png", width="40vw")
    _cols = mo.hstack([_lc, _rc], justify="space-between", gap=5, widths=[1, 1])
    mo.vstack([_title, _cols], align="start", justify="start", gap="5")
    return


@app.cell
def _(mo):
    _title = mo.md("## Other features/tools not covered in this webinar")
    _lc = mo.md(
        r"""
    - parallelization implementation and examples with `dask` (https://github.com/NREL/solar-data-tools/tree/main/sdt_dask)
    - best irriance sensor identification
    - more advanced uses of quantile fitting and transformation (beyond clear sky modeling)
    - open PR for fleet-scale partial outage detection

    &nbsp;

    - new features always underdevelopment, welcome ideas and code contributions! (Bundt cakes recently added by Stanford grad student)
    """
    )
    _rc = mo.image("assets/bundt.gif", width="50vw")
    _cols = mo.hstack([_lc, _rc], justify="space-between", gap=5, widths=[1, 1])
    mo.vstack([_title, _cols], align="start", justify="start", gap="5")
    return


@app.cell
def _(mo):
    _title = mo.md("## Conclusions and next steps")
    _lc = mo.md(
        r"""
    - SDT provides a feature-rich package for hands-on PV data exploration and large-scale pipeline automation
    - ML/SP algorithms execute in seconds to minutes on a standard laptop for a single data stream
    - just a few lines of code to analyze any type of system from small rooftop to large tracking

    &nbsp;

    - **we want to hear from you!**
    - please fill out the feedback form that will be shared in the chat and by email after the webinar
    - also please reach out if you have questions, comments, or just want help with implementation
    - we are interested in helping companies scale to very large pipelines

    &nbsp;

    - **next up:** PV Fleets and large data pipelines
    - **later on:** open-source community building

    """
    )
    _acknowledgement = "This work was authored in part by NREL for the U.S. Department of Energy (DOE), operated under Contract No. DE-AC36-08GO28308. Funding provided by the U.S. Department of Energy’s Office of Energy Efficiency and Renewable Energy (EERE) under the Solar Energy Technologies Office Award Number 38529."
    mo.vstack(
        [
            _title,
            _lc,
            mo.hstack(
                [
                    mo.image(
                        src="assets/SDT_v1_secondary_blue_text.png", width="10vw"
                    ).center(),
                    _acknowledgement,
                ],
                align="center",
                widths=[1, 3],
                gap=1,
            ),
        ],
        gap=5,
    )
    return


if __name__ == "__main__":
    app.run()
