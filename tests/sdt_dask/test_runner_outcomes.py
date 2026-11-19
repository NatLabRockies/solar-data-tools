import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("dask.distributed")

from sdt_dask.dask_tool import runner
from solardatatools import DataHandler
from solardatatools.algorithms import SunriseSunset


def test_runner_retains_partial_results_and_serializes_outcomes(tmp_path, monkeypatch):
    index = pd.date_range("2024-01-01", periods=72, freq="h")
    power = np.tile(np.r_[np.zeros(6), np.arange(1.0, 13), np.zeros(6)], 3)

    class DataPlug:
        def get_data(self, key):
            if key == ("missing",):
                raise FileNotFoundError("data missing")
            return pd.DataFrame({"power": power}, index=index)

    def failed_daylight(*args, **kwargs):
        raise RuntimeError("daylight solve failed")

    class Client:
        def compute(self, result):
            return SimpleNamespace(
                result=lambda: result.compute(scheduler="synchronous")
            )

        def shutdown(self):
            pass

    monkeypatch.setattr(SunriseSunset, "run_optimizer", failed_daylight)
    monkeypatch.setattr(runner, "performance_report", lambda path: nullcontext())
    job = runner.Runner(Client(), str(tmp_path))
    job.set_up([("available",), ("missing",)], DataPlug(), fix_shifts=False)
    results = job.df_results.compute(scheduler="synchronous").set_index("key_field_0")
    row = results.loc["available"]
    assert row["capacity"] > 0
    assert "daylight solve failed" in row["run_pipeline error"]
    assert row["run_pipeline_report error"] == "No error"
    assert row["stage outcomes"]["time shift"]["status"] == "unrequested"
    assert row["stage outcomes"]["loss analysis"]["status"] == "skipped"
    assert "one year" in row["run_loss_analysis error"]
    assert results.loc["missing", "get_data error"] == "data missing"
    assert (
        results.loc["missing", "stage outcomes"]["data loading"]["status"] == "failed"
    )

    job.get_result()
    csv = pd.read_csv(next(tmp_path.glob("*-summary_report.csv"))).set_index(
        "key_field_0"
    )
    assert (
        json.loads(csv.loc["available", "stage outcomes"])["provisional daylight"][
            "status"
        ]
        == "failed"
    )


def test_runner_retains_report_after_input_exception(monkeypatch):
    index = pd.date_range("2024-01-01", periods=48, freq="h")
    source = pd.DataFrame({"power": np.ones(len(index))}, index=index)
    job = runner.Runner(client=None)
    job.set_up(
        [("site",)],
        SimpleNamespace(get_data=lambda key: source),
        sampling_interval="7h",
    )
    row = job.df_results.compute(scheduler="synchronous").iloc[0]
    assert row["stage outcomes"]["observation preparation"]["status"] == "failed"
    assert row["run_pipeline_report error"] == "No error"
    assert "divide a 24-hour day" in row["run_pipeline error"]


def test_runner_attempts_independent_loss_analysis_after_partial_failure(monkeypatch):
    index = pd.date_range("2024-01-01", periods=24 * 366, freq="h")
    source = pd.DataFrame({"power": np.ones(len(index))}, index=index)
    calls = []

    def daylight(analysis, data, **kwargs):
        analysis.sunup_mask_estimated = np.ones_like(data, dtype=bool)
        analysis.sunrise_estimates = np.zeros(data.shape[1])
        analysis.sunset_estimates = np.full(data.shape[1], 24)

    def fail_final(*args, **kwargs):
        raise RuntimeError("refresh failed")

    def loss(handler, **kwargs):
        calls.append(handler.num_days)
        handler._record_stage("loss analysis", "success")
        handler.loss_analysis = SimpleNamespace(
            report=lambda: {"total energy loss [kWh]": -1.0}
        )

    monkeypatch.setattr(SunriseSunset, "run_optimizer", daylight)
    monkeypatch.setattr(SunriseSunset, "calculate_times", fail_final)
    monkeypatch.setattr(
        DataHandler, "_score_and_analyze_days", lambda self, **kwargs: None
    )
    monkeypatch.setattr(DataHandler, "run_loss_factor_analysis", loss)
    job = runner.Runner(client=None)
    job.set_up(
        [("site",)], SimpleNamespace(get_data=lambda key: source), fix_shifts=False
    )
    row = job.df_results.compute(scheduler="synchronous").iloc[0]
    assert calls == [366]
    assert row["stage outcomes"]["final daylight"]["status"] == "failed"
    assert row["stage outcomes"]["loss analysis"]["status"] == "success"
    assert row["total energy loss [kWh]"] == -1.0
