from carla.evaluation.api import Evaluation
from carla.recourse_methods.processing import merge_default_parameters
from carla.evaluation import remove_nans
from cote.vae import VariationalAutoencoder
import numpy as np
import pandas as pd
import torch

class VAEBenchmark(Evaluation):
    """
    Computes the euclidean distance between the latent spaces between the counterfactuals and factuals
    """
    _DEFAULT_HYPERPARAMS = {
        "myvae_params": {
            'input_dim': None,
            'kld_weight': 0.0025,
            'layers': [20, 10],
            'latent_dim': 7,
            'hidden_activation': 'relu',
            'dropout': 0.2,
            'batch_norm': True,
            'batch_size': 64,
            'epochs': 20,
            'learning_rate': 0.001,
            'weight_decay': 0.0,
            'cuda': False,
            'verbose': True,
            'train': True,
            'save_dir': './vae_model/'
        }
    }
    def __init__(self, mlmodel, hyperparameters):
        hyperparameters = merge_default_parameters(hyperparameters, self._DEFAULT_HYPERPARAMS)
        super().__init__(mlmodel, hyperparameters)
        hyperparameters['myvae_params']['input_dim'] = len(mlmodel.feature_input_order)
        self.columns = ["VAE-Euclidean-Distance"]
        self._initialize_vae(vae_params = hyperparameters['myvae_params'])

    def _initialize_vae(self, vae_params):
        data_name = self.mlmodel.data.name
        mutable_list = self.mlmodel.get_mutable_mask() 
        self.vae = VariationalAutoencoder(
            data_name, vae_params["layers"], mutable_list, vae_params
        )

        self.vae.fit(xtrain=self.mlmodel.data.df[self.mlmodel.feature_input_order])
    
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

        # Get the VAE encodings for the factuals and counterfactuals
        arr_f = torch.FloatTensor(arr_f)
        arr_cf = torch.FloatTensor(arr_cf)

        vae_encodings_f = self.vae.encode(arr_f)[0].detach().numpy()
        vae_encodings_cf = self.vae.encode(arr_cf)[0].detach().numpy()

        # Get the VAE distances between the factuals and counterfactuals
        vae_distances = self.get_distances(vae_encodings_f, vae_encodings_cf)
        
        return pd.DataFrame(vae_distances, columns=self.columns)