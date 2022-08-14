import pandas as pd

from carla.evaluation.api import Evaluation


class AvgTime(Evaluation):
    """
    Computes average time for generated counterfactual
    """

    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        self.time = hyperparameters["time"]
        self.columns = ["timers"]

    def get_evaluation(
        self, factuals: pd.DataFrame, counterfactuals: pd.DataFrame
    ) -> pd.DataFrame:
        # Return the list of self.time as a dataframe with columns self.columns
        return pd.DataFrame(self.time, columns=self.columns)
