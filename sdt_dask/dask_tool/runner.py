"""

This module provides a class to run the SolarDataTools pipeline on a Dask cluster.
It takes a data plug and a dask client as input and runs the pipeline on the data plug

See the README and tool_demo_SDTDask.ipynb for more information

"""

import json
import os
import pandas as pd
from dask import delayed
from time import perf_counter, strftime
from dask.distributed import performance_report
from solardatatools import DataHandler
from solardatatools._reporting import format_outcomes, pipeline_error


class Runner:
    """A class to run the SolarDataTools pipeline on a Dask cluster.
    Will handle invalid data keys and failed datasets.

    :param keys:
        data_plug (:obj:`DataPlug`): The data plug object.
        client (:obj:`Client`): The Dask client object.
        output_path (str): The path to save the results.
    """

    def __init__(self, client, output_path="../results/"):
        self.client = client
        self.output_path = output_path

    def set_up(self, KEYS, data_plug, **kwargs):
        """function to set up the pipeline on the data plug

        Call run_pipeline functions in a for loop over the keys
        and collect results in a DataFrame

        :param keys:
            KEYS (list): List of tuples
            **kwargs: Optional parameters.

        """

        def get_data(key):
            """Creates a dataframe for a key that includes the errors for the
            functions performed on the file and its datahandler.

            :param key: The key combination of the file
            :type key: tuple
            :return: Returns the dataframe for the key and its datahandler
            :rtype: tuple"""

            # Addition of function error status for key
            errors = {
                "get_data error": ["No error"],
                "run_pipeline error": ["No error"],
                "run_pipeline_report error": ["No error"],
                "run_loss_analysis error": ["No error"],
                "run_loss_analysis_report error": ["No error"],
            }

            # Creates a key dictionary for the key combinations and the sensor
            # information
            key_dict = {}
            for i in range(len(key)):
                key_dict[f"key_field_{i}"] = key[i]

            # Combines the key and errors into a dictionary and generates a
            # dataframe
            data_dict = {**key_dict, **errors}
            data_df = pd.DataFrame.from_dict(data_dict)

            started = perf_counter()
            try:
                # Reads CSV file and creates dataframe, Here the function reads
                # data into pandas and python instead of using dask to store the
                # data in memory for further computation **.
                df = data_plug.get_data(key)
                dh = DataHandler(df)
                return (data_df, dh)
            except Exception as e:
                data_df["get_data error"] = str(e) or type(e).__name__
                data_df["stage outcomes"] = [
                    {
                        "data loading": {
                            "status": "failed",
                            "reason": str(e) or type(e).__name__,
                            "elapsed seconds": perf_counter() - started,
                        }
                    }
                ]
                return (data_df,)

        def run_pipeline(data_tuple, **options):
            data_df = data_tuple[0]
            if len(data_tuple) == 1:
                reason = "Unavailable because data loading failed."
                for column in (
                    "run_pipeline error",
                    "run_pipeline_report error",
                    "run_loss_analysis error",
                    "run_loss_analysis_report error",
                ):
                    data_df[column] = reason
                return data_tuple

            datahandler = data_tuple[1]
            try:
                datahandler.run_pipeline(**options)
            except Exception as error:
                data_df["run_pipeline error"] = str(error) or type(error).__name__
            return data_df, datahandler

        def run_loss_analysis(data_tuple):
            data_df = data_tuple[0]
            if len(data_tuple) == 1:
                return data_df

            datahandler = data_tuple[1]
            try:
                report = datahandler.report(return_values=True, verbose=False)
                outcomes = report.pop("stage outcomes")
            except Exception as error:
                data_df["run_pipeline_report error"] = str(error)
                data_df["run_loss_analysis error"] = "Pipeline report unavailable."
                data_df["run_loss_analysis_report error"] = "Loss analysis unavailable."
                return data_df

            data_df = data_df.assign(**report)
            summary = pipeline_error(outcomes)
            if summary != "No error":
                data_df["run_pipeline error"] = summary
            data_df["runtimes"] = datahandler.total_time

            if datahandler.num_days is not None and datahandler.num_days <= 365:
                outcomes["loss analysis"] = {
                    "status": "skipped",
                    "reason": "Dask loss analysis requires more than one year of data.",
                    "elapsed seconds": 0.0,
                }
            else:
                try:
                    datahandler.run_loss_factor_analysis(verbose=False)
                except Exception as error:
                    data_df["run_loss_analysis error"] = (
                        str(error) or type(error).__name__
                    )
                outcomes["loss analysis"] = datahandler.report(
                    return_values=True, verbose=False
                )["stage outcomes"]["loss analysis"]

            loss = outcomes["loss analysis"]
            if loss["status"] in {"failed", "skipped"}:
                data_df["run_loss_analysis error"] = format_outcomes(
                    {"loss analysis": loss}
                )
                data_df["run_loss_analysis_report error"] = "Loss analysis unavailable."
            elif datahandler.loss_analysis is not None:
                try:
                    data_df = data_df.assign(**datahandler.loss_analysis.report())
                except Exception as error:
                    data_df["run_loss_analysis_report error"] = str(error)
            data_df["stage outcomes"] = [outcomes]
            return data_df

        results = []

        # For larger number of files it is recommended to use dask collections
        # instead of a for loop **
        # Reference:
        #   https://docs.dask.org/en/latest/delayed-best-practices.html#avoid-too-many-tasks
        options = {"fix_shifts": True, "verbose": False, **kwargs}
        for key in KEYS:
            data_tuple_0 = delayed(get_data)(key)
            data_tuple_1 = delayed(run_pipeline)(data_tuple_0, **options)

            result_df = delayed(run_loss_analysis)(data_tuple_1)
            results.append(result_df)

        self.df_results = delayed(pd.concat)(results)

    def visualize(self, filename="sdt_graph.png", **kwargs):
        # visualize the pipeline, user should have graphviz installed
        self.df_results.visualize(filename, **kwargs)

    def get_result(
        self, dask_report="dask-report.html", summary_report="summary_report.csv"
    ):
        # test if the filepath exist, if not create it
        time_stamp = strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Compute tasks on cluster and save results
        with performance_report(
            self.output_path + "/" + f"{time_stamp}-" + dask_report
        ):
            summary_table = self.client.compute(self.df_results)
            df = summary_table.result()
            if "stage outcomes" in df:
                df = df.assign(
                    **{
                        "stage outcomes": df["stage outcomes"].map(
                            lambda outcomes: json.dumps(outcomes)
                            if isinstance(outcomes, dict)
                            else None
                        )
                    }
                )
            df.to_csv(self.output_path + "/" + f"{time_stamp}-" + summary_report)

        self.client.shutdown()
