"""Exercise the installed solar-data-tools wheel with its base dependencies."""

import sys
from importlib import import_module, metadata, resources
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solardatatools import DataHandler, __version__, make_2d, standardize_time_axis
from solardatatools.signal_decompositions import l2_l1d2_constrained


def check_distribution(checkout):
    distribution = metadata.distribution("solar-data-tools")
    installed_files = {
        Path(distribution.locate_file(file)).resolve() for file in distribution.files
    }
    assert __version__ == distribution.version, (__version__, distribution.version)
    for name in ("solardatatools", "pvsystemprofiler", "anomalydetector", "sdt_dask"):
        origin = Path(import_module(name).__file__).resolve()
        assert origin in installed_files and not origin.is_relative_to(checkout), origin


def check_resources():
    fixtures = resources.files("solardatatools").joinpath("fixtures")
    for name in ("clear_PCA.txt", "Q.txt", "eigvals.txt", "mu.txt"):
        with fixtures.joinpath(name).open() as stream:
            data = np.loadtxt(stream)
            assert data.size, name
    with fixtures.joinpath("transformed_data_corpus.zip").open("rb") as stream:
        with ZipFile(stream) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            assert members, "Expected a packaged shading corpus."
            with archive.open(members[0]) as data:
                assert data.read(1), "The packaged shading corpus is empty."


def check_transforms_and_plot():
    index = pd.date_range("2024-01-01", periods=192, freq="15min")
    values = np.arange(len(index), dtype=float)
    values[42] = np.nan
    source = pd.DataFrame({"power": values}, index=index)

    aligned, _ = standardize_time_axis(source, verbose=False)
    np.testing.assert_array_equal(aligned.index, index)
    matrix, dates = make_2d(aligned, key="power", return_day_axis=True)
    assert matrix.shape == (96, 2), matrix.shape
    np.testing.assert_array_equal(dates, index[::96])
    np.testing.assert_array_equal(matrix, values.reshape(96, 2, order="F"))

    handler = DataHandler(raw_data_matrix=matrix)
    figure = handler.plot_heatmap(scale_to_kw=False)
    try:
        image = Path("raw-heatmap.png")
        figure.savefig(image)
        assert image.stat().st_size > 0
    finally:
        plt.close(figure)


def check_public_solve():
    ramp = np.linspace(0, 1, 21)
    # Use the public result convention until SD-01 introduces named components.
    _, fitted, _ = l2_l1d2_constrained(ramp, solver="CLARABEL")
    assert fitted.shape == ramp.shape, fitted.shape
    assert np.isfinite(fitted).all()
    np.testing.assert_allclose(fitted, ramp, atol=1e-4, rtol=0)


if __name__ == "__main__":
    check_distribution(Path(sys.argv[1]).resolve())
    check_resources()
    check_transforms_and_plot()
    check_public_solve()
    print("Installed-wheel smoke checks passed.")
