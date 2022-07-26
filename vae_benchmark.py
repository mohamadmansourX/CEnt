from carla.recourse_methods.autoencoder import (
    VariationalAutoencoder,
    train_variational_autoencoder,
)
from carla.evaluation.api import Evaluation
from carla.recourse_methods.processing import merge_default_parameters
from carla.evaluation import remove_nans
import numpy as np
import pandas as pd


class VAEBenchmark(Evaluation):
    """
    Computes the euclidean distance between the latent spaces between the counterfactuals and factuals
    """
    _DEFAULT_HYPERPARAMS = {
          "layers": [512, 250, 32],
          "train": True,
          "lambda_reg": 1e-6,
          "kl_weight": 0.3,
          "epochs": 15,
          "lr": 1e-3,
          "batch_size": 64,
          "mutables": []
      }
    def __init__(self, mlmodel, hyperparameters):
        hyperparameters = merge_default_parameters(hyperparameters, self._DEFAULT_HYPERPARAMS)
        super().__init__(mlmodel, hyperparameters)
        self.columns = ["VAE-Euclidean-Distance"]
        self._initialize_vae(vae_params = hyperparameters)
    
    def _initialize_vae(self, vae_params):
        data_name = self.mlmodel.data.name
        self.vae = VariationalAutoencoder(
            data_name, vae_params["layers"], vae_params["mutables"]
        )

        self.vae.fit(xtrain=self.mlmodel.data.df[self.mlmodel.feature_input_order],
                kl_weight=vae_params["kl_weight"],
                lambda_reg=vae_params["lambda_reg"],
                epochs=vae_params["epochs"],
                lr=vae_params["lr"],
                batch_size=vae_params["batch_size"],
            )
    
    def get_minkowski_distance(self, v1s, v2s, p = 2):
        '''
        Compute the Minkowski distance between two vectors
        p should be a positive integer
        if p = 1, this is equivalent to the Manhattan distance
        if p = 2, this is equivalent to the Euclidean distance
        '''
        return np.linalg.norm(v1s - v2s, ord=p, axis = 1)

    def get_distances(self, arr_f, arr_cf):
        '''
        Compute the VAE distance between two vectors
        '''
        vae_distances = self.get_minkowski_distance(arr_f, arr_cf, p = 2)
        return vae_distances

    def get_evaluation(self, factuals, counterfactuals):
        # only keep the rows for which counterfactuals could be found
        counterfactuals_without_nans, factuals_without_nans = remove_nans(
            counterfactuals, factuals
        )

        # return empty dataframe if no successful counterfactuals
        if counterfactuals_without_nans.empty:
            return pd.DataFrame(columns=self.columns)
        

        arr_f = self.mlmodel.get_ordered_features(factuals_without_nans).to_numpy()
        arr_cf = self.mlmodel.get_ordered_features(
            counterfactuals_without_nans
        ).to_numpy()

        vae_distances = self.get_distances(arr_f, arr_cf)
        
        return pd.DataFrame(vae_distances, columns=self.columns)