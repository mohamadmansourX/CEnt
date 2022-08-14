import timeit
from typing import List

import pandas as pd

from carla.evaluation.api import Evaluation
from carla.models.api import MLModel
from carla.recourse_methods.api import RecourseMethod


class Benchmark:
    """
    The benchmarking class contains all measurements.
    It is possible to run only individual evaluation metrics or all via one single call.

    For every given factual, the benchmark object will generate one counterfactual example with
    the given recourse method.

    Parameters
    ----------
    mlmodel: carla.models.MLModel
        Black Box model we want to explain.
    recourse_method: carla.recourse_methods.RecourseMethod
        Recourse method we want to benchmark.
    factuals: pd.DataFrame
        Instances for which we want to find counterfactuals.
    """

    def __init__(
        self,
        mlmodel: MLModel,
        recourse_method: RecourseMethod,
        factuals: pd.DataFrame,
    ) -> None:
        colls = list(mlmodel.data.df.columns)
        colls.remove(mlmodel.data.target)
        self.mlmodel = mlmodel
        self._recourse_method = recourse_method
        self._factuals2 = factuals[colls].astype(float).copy()
        self._factuals = self.mlmodel.get_ordered_features(factuals.copy()).astype(float)
        timers_per_row = []
        counterfacts = []
        for ind in self._factuals.index:
            # Format self._factuals.loc[ind] as dataframe
            # Generate counterfactual
            start_timer = timeit.default_timer()
            counterfactual = self._recourse_method.get_counterfactuals(self._factuals.loc[[ind]]).astype(float)
            # Append time and counterfactual to list
            timers_per_row.append(timeit.default_timer() - start_timer)
            counterfacts.append(counterfactual)
        # concatenate all 
        self._counterfactuals = pd.concat(counterfacts, axis=0)
        # Copy columns that are not in the counterfactuals
        self._counterfactuals2 = self._counterfactuals.copy()
        for col in self._factuals2.columns:
            if col not in self._counterfactuals2.columns:
                self._counterfactuals2[col] = self._factuals2[col]
        self.timer = timers_per_row

    def run_benchmark(self, measures: List[Evaluation]) -> pd.DataFrame:
        """
        Runs every measurement and returns every value as dict.

        Parameters
        ----------
        measures : List[Evaluation]
            List of Evaluation measures that will be computed.

        Returns
        -------
        pd.DataFrame
        """
        pipeline = []
        for measure in measures:
            if 'Constraint_Violation' in measure.columns:
                pipeline.append(measure.get_evaluation(
                    counterfactuals=self._counterfactuals2, factuals=self._factuals2
                    ))
            elif "VAE-Euclidean-Distance" in measure.columns:
                pipeline.append(measure.get_evaluation(
                    counterfactuals=self._counterfactuals2, factuals=self._factuals2
                    ))
            else:
                pipeline.append(measure.get_evaluation(
                    counterfactuals=self._counterfactuals, factuals=self._factuals
                    ))
                
        output = pd.concat(pipeline, axis=1)

        return output
