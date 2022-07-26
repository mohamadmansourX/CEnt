from tqdm.notebook import tqdm
from pynndescent import NNDescent
import enum
from typing import Dict, List, Tuple, Union
import pandas as pd
from carla import RecourseMethod
from carla.data.api import data, Data
from carla.models.api import MLModel
from carla.recourse_methods.autoencoder import (
    VariationalAutoencoder,
    train_variational_autoencoder,
)
from carla.recourse_methods.processing import (
    check_counterfactuals,
    merge_default_parameters,
    reconstruct_encoding_constraints,
)
from cote.TreeLeaf import TreeLeafs
# For Descision Tree implementation
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from carla import Benchmark
from carla.recourse_methods import Dice, Face
import numpy as np
import random
from sklearn.model_selection import train_test_split
from copy import deepcopy
from carla import MLModelCatalog
from carla.data.catalog import OnlineCatalog
from carla.recourse_methods import GrowingSpheres
from sklearn.model_selection import GridSearchCV, train_test_split
tqdm.pandas()


class TreeBasedContrastiveExplanation(RecourseMethod):
    '''
    Decision Tree Based contrastive explanations
    '''
    _DEFAULT_HYPERPARAMS = {
      "data_name": None,
      "n_search_samples": 300,
      "p_norm": 1,
      "step": 0.1,
      "max_iter": 1000,
      "clamp": True,
      "target_class": [0, 1],
      "binary_cat_features": True,
      "vae_params": {
          "layers": [50, 10],
          "train": True,
          "lambda_reg": 1e-6,
          "kl_weight": 0.3,
          "epochs": 5,
          "lr": 1e-3,
          "batch_size": 32,
      },
      "tree_params": {
          "min_entries_per_label": 1000,
          "grid_search_jobs": -1,
          "min_weight_gini": 100, # set to 0.5 since here both class have same prob
          "max_search": 500,
          "grid_search": {
                "splitter": ["best"],
                "criterion": ["gini"],
                "max_depth": [6],
                "min_samples_split": [2],
                "min_samples_leaf": [1],
                "max_features": [None] #Note changing this will result in removing features that we might want to keep
          }
      }

    }

    def __init__(self, dataset:Data, mlmodel: MLModel, hyperparams: Dict, data_catalog: Dict, distance_metric ='euclidean'):
        super().__init__(mlmodel)
        self.distance_metric = distance_metric
        # Construct catalog
        self.data_catalog = data_catalog
        # Construct mlmodel
        self.mlmodel = mlmodel
        # Construct the hyperparameters
        self.hyperparams = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)
        # Construct the VAE
        # self.vae = TEMP_VAE
        self.vae = self.load_vae(dataset, self.hyperparams["vae_params"], mlmodel, mlmodel.data.name)
        # Construct the dataframe with encodings
        self.dataset = dataset.df
        self.dataset['VAE_ENCODED'] = self.get_encodeings(self.dataset.copy())
        # Load Grid Parameters
        self.hyperparams["tree_params"]["grid_search"] = self.optimize_grid(self.hyperparams["tree_params"]["grid_search"], self.dataset)
        ## These are added to optimize neighbor sampling for DT which used to take ~0.4 seconds and now will be 
        # NNDescent
        self.data_indexes_m = self.dataset.index
        self.set_distance_metric_initialize_nn(self.distance_metric)
    def set_distance_metric_initialize_nn(self, distance_metric):
        self.distance_metric = distance_metric
        self.nnd = NNDescent(np.array(self.dataset["VAE_ENCODED"].values.tolist()), metric=self.distance_metric,random_state=42)
        self.nnd.prepare()
    def set_model(self,mlmodel):
        self._mlmodel = mlmodel
        self.mlmodel = mlmodel
    def optimize_grid(self, grid_search, df):
        """
        Optimize the grid search parameters
        #@TODO: make it on chunkc of the dataframe and return the top
        #@TODO: the chunkcs in order of encodings
        """
        # Define Decision Tree Classifier
        dec_tree = DecisionTreeClassifier(random_state=0)
        # Define Grid Search
        grid_search = GridSearchCV(dec_tree, grid_search, cv=5, n_jobs=self.hyperparams["tree_params"]["grid_search_jobs"])
        target_values = df[self._mlmodel.data.target]
        train_features = df[self._mlmodel.feature_input_order]
        # Fit the Grid Search
        grid_search.fit(train_features, target_values)
        # Return the best parameters as the format of grid_search
        print(grid_search.best_params_)
        return grid_search.best_params_

    def load_vae(
        self, data: pd.DataFrame, vae_params: Dict, mlmodel: MLModel, data_name: str
    ) -> VariationalAutoencoder:
        '''
        Load and train the VAE if needed
        '''
        mutable_list = mlmodel.get_mutable_mask() 
        # Change all False to True in mutable_list
        mutable_list = np.array([True for x in mutable_list])
        generative_model = VariationalAutoencoder(
            data_name, vae_params["layers"], mutable_list
        )
        print(generative_model)
        if vae_params["train"]:
            generative_model.fit(
                xtrain=data.df[mlmodel.feature_input_order],
                kl_weight=vae_params["kl_weight"],
                lambda_reg=vae_params["lambda_reg"],
                epochs=vae_params["epochs"],
                lr=vae_params["lr"],
                batch_size=vae_params["batch_size"],
            )
        else:
            try:
                generative_model.load(vae_params["layers"][0])
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Loading of Autoencoder failed. {}".format(str(exc))
                )

        return generative_model

    def get_counterfactuals(self, factuals: pd.DataFrame):
        '''
        this property is responsible to generate and output
        encoded and scaled counterfactual examples
        as pandas DataFrames
        '''
        # Get the encoded features of factuals
        factuals["VAE_ENCODED"] = self.get_encodeings(factuals)
        # Get the counterfactuals
        # find counterfactuals
        counter_factuals = factuals.apply(
            lambda x: self.tree_based_search(x), axis=1, raw=False
        )
        # counter_factuals = [self.tree_based_search(row) for __,row in factuals.iterrows()]
        # Concatenate the counterfactuals to a single dataframe
        # counter_factuals is a list of rows
        self.counter_factuals = counter_factuals
        #counter_factuals = check_counterfactuals(self._mlmodel, counter_factuals)
        # Return the counterfactuals
        return counter_factuals[self._mlmodel.feature_input_order]

    def get_encodeings(self, data: pd.DataFrame):
        '''
        This method is responsible to append the encoded features
        to the dataframe
        '''
        # Fix DataFrame to be able to feed to the VAE
        input_data = data.copy()[self._mlmodel.feature_input_order]
        input_data = torch.FloatTensor(input_data.values)
        # Get the encoded features
        encoded_values = self.vae.encode(input_data)[0].detach().numpy()
        encoded_values = [i for i in encoded_values]
        return encoded_values

    def distance_get(self, x,factuals):
        return np.square((x - factuals)).sum()

    def get_nearest_neighbors_thershold(self, copy_data, label_threshold):
        '''
        This method is responsible to get the nearest neighbors of a given threshold
        using the VAE and minimum threshold per label
        '''
        # Find the index of the 100th instance of each class
        id_100th_class_0 = copy_data[copy_data[self._mlmodel.data.target] == 0].index[label_threshold-1]
        id_100th_class_1 = copy_data[copy_data[self._mlmodel.data.target] == 1].index[label_threshold-1]
        # Get the maximum id
        max_id = max(id_100th_class_0, id_100th_class_1)
        # Return the nearest neighbors of the 100th instance of each class
        return copy_data.head(max_id)
    
    def decision_tree(self, nearest_neighbors):
        '''
        This method is responsible to create a decision tree
        using the nearest neighbors of the 100th instance of each class
        '''
        target_values = nearest_neighbors[self._mlmodel.data.target]
        train_features = nearest_neighbors[self._mlmodel.feature_input_order]
        # Create the decision tree
        clf = DecisionTreeClassifier(random_state=0 , max_depth=self.hyperparams["tree_params"]['grid_search']["max_depth"], 
                                    min_samples_split=self.hyperparams["tree_params"]['grid_search']["min_samples_split"], 
                                    min_samples_leaf=self.hyperparams["tree_params"]['grid_search']["min_samples_leaf"], 
                                    max_features=self.hyperparams["tree_params"]['grid_search']["max_features"])
        # Define the grid search
        #grid_search = GridSearchCV(clf, self.hyperparams["tree_params"]["grid_search"], cv=5, verbose=0, refit=True, n_jobs=self.hyperparams["tree_params"]["grid_search_jobs"])
        # Fit the grid search evaluate on X_test and y_test then refit best model on the whole dataset
        #grid_search.fit(train_features, target_values)
        # Return the best model
        #return grid_search.best_estimator_
        clf.fit(train_features, target_values)
        return clf

    def tree_based_search(self, factual):
        '''
        This method is responsible to get the counterfactual of a given targeted_encoding
        '''
        copy_data = self.dataset.copy()
        ## Get distances from data to this encoding
        #time1 = time.time()
        #copy_data["distance"] = copy_data["VAE_ENCODED"].apply(lambda x: self.distance_get(x, factual["VAE_ENCODED"]))
        #print("Old Method Took: {:.3f} seconds".format(time.time()-time1))
        ### Use the KD tree to query the 1000 neighbors of data_indexes_1 from factual["VAE_ENCODED"]
        ##TODO SHIFT TO pynndescent
        #index_neighbors_1 = self.kdtree_1.query(factual["VAE_ENCODED"], k=self.hyperparams["tree_params"]["min_entries_per_label"])[1]
        ### index_neighbors_1 are the locations of indexes in self.data_indexes_1
        ### Use index_neighbors_1 to get the data_indexes_1
        #datata_index_1 = self.data_indexes_1[index_neighbors_1].tolist()
        #index_neighbors_0 = self.kdtree_0.query(factual["VAE_ENCODED"], k=self.hyperparams["tree_params"]["min_entries_per_label"])[1]
        #datata_index_0 = self.data_indexes_0[index_neighbors_0].tolist()
        ### combine the data_indexes_1 and data_indexes_0 lists
        #datata_index_0.extend(datata_index_1)
        ## Get the data from the indexes
        # MAIN KDTREE
        #index_neighbors_0 = self.kdtree_m.query(factual["VAE_ENCODED"], k=self.hyperparams["tree_params"]["min_entries_per_label"]*2.5)[1].tolist()
        # NNDescent with manhatan
        index_neighbors_0 = self.nnd.query(np.array([factual["VAE_ENCODED"].tolist()]), k=self.hyperparams["tree_params"]["min_entries_per_label"]*2.5)[0][0].tolist()
        datata_index_0 = self.data_indexes_m[index_neighbors_0].tolist()
        nearest_neighbors = copy_data.loc[datata_index_0]
        #datata_index_1 = np.random.choice(self.data_indexes_1, self.hyperparams["tree_params"]["min_entries_per_label"], replace=False).tolist()
        #datata_index_0 = np.random.choice(self.data_indexes_0, self.hyperparams["tree_params"]["min_entries_per_label"], replace=False).tolist()
        #datata_index_0.extend(datata_index_1)
        #nearest_neighbors = copy_data.loc[datata_index_0]        
        ## Sort the dataframe by distance
        #time1 = time.time()
        #copy_data = copy_data.sort_values(by="distance")
        #print("Sort Took: {:.3f} seconds".format(time.time()-time1))
        ## Reset the index
        #copy_data = copy_data.reset_index(drop=True)
        ## Get the nearest neighbors of the targeted encoding
        #time1 = time.time()
        #nearest_neighbors = self.get_nearest_neighbors_thershold(copy_data, label_threshold=self.hyperparams["tree_params"]["min_entries_per_label"])
        #print("Get Neighbors Took: {:.3f} seconds".format(time.time()-time1))
        # Get the tree
        tree = self.decision_tree(nearest_neighbors)
        self.mtree = tree
        # Get the leaf nodes
        leaf_nodes = TreeLeafs(tree.tree_, self._mlmodel.feature_input_order).leafs_nodes.copy()
        # leaf_nodes is list of classes LeafNode
        # Get the leaf node where the targeted encoding is located
        leaf_node_n_i = -1
        for leaf_i in range(len(leaf_nodes)):
            if leaf_nodes[leaf_i].check_point(factual):
                leaf_node_n_i = leaf_i
                break
        self.mleaf_node_n_i = leaf_node_n_i
        self.mfactual = factual
        # assert if the leaf node is not found
        assert leaf_node_n_i != -1, "Leaf node not found"
        # For now change leafnode label to the item label
        # #assert if leaf_node_n.label is not the same as the factual label
        # assert leaf_node_n.label == factual[self._mlmodel.data.target], "Leaf node label {} is not the same as the factual label {}".format(leaf_node_n.label, factual[self._mlmodel.data.target])
        if leaf_nodes[leaf_node_n_i].label != factual[self._mlmodel.data.target]:
          #print("Leaf Node {} flipped node label {} to match the factual entry {}".format(leaf_node_n_i,
          #                                                                          leaf_nodes[leaf_node_n_i].label,
          #                                                                          factual[self._mlmodel.data.target]))
          leaf_nodes[leaf_node_n_i].label = factual[self._mlmodel.data.target]
        leaf_node_n = leaf_nodes[leaf_node_n_i]
        # Get all leafnodes with label!= leaf_node_n.label and Sort leaf nodes by distance
        leaf_nodes_with_label = [leaf_n for leaf_n in leaf_nodes if leaf_n.label != leaf_node_n.label]
        # Check if leaf_nodes_with_label is empty
        if len(leaf_nodes_with_label) == 0:
            print("No leaf node with label {}".format(leaf_node_n.label))
            factual_ret = factual
            factual_ret[self._mlmodel.feature_input_order] = np.nan
            #print("returned")
            return factual_ret[self._mlmodel.feature_input_order]
        # Sort leaf nodes by distance
        leaf_nodes_with_label = sorted(leaf_nodes_with_label, key=lambda x: len(leaf_node_n.compare_node(x)))
        # Get the counterfactual
        returned_neighbor = -1
        counter_taregt = factual[self._mlmodel.data.target]*-1 +1
        #print("Searching for Neighbor....")
        # print("Start with option A: {}".format(nearest_leaf_node))
        #print(second_nearest_node)
        # If len of leaf_nodes_with_label is 1, the all the max_search on the nearest_leaf_node
        # If len of leaf_nodes_with_label is 2, the max_search/7 on the nearest_leaf_node and max_search/3 on the second_nearest_node
        # If len of leaf_nodes_with_label is 3, the max_search/5 on the nearest_leaf_node and max_search/3 on the second_nearest_node and max_search/2 on the third_nearest_node
        if len(leaf_nodes_with_label) == 1:
            max_searchs = [self.hyperparams["tree_params"]["max_search"]]
        elif len(leaf_nodes_with_label) == 2:
            max_searchs = [self.hyperparams["tree_params"]["max_search"]*0.8, self.hyperparams["tree_params"]["max_search"]*0.2]
        else:
            max_searchs = [self.hyperparams["tree_params"]["max_search"]*0.7, self.hyperparams["tree_params"]["max_search"]*0.2, self.hyperparams["tree_params"]["max_search"]*0.1]
        # map max_search to int values while rounding up to the nearest int
        max_searchs = [int(round(x)) for x in max_searchs]
        # Loop over max_search
        for rank_node, max_search_i in enumerate(max_searchs):
            number_searchs = 0
            nearest_leaf_node = leaf_nodes_with_label[rank_node]
            #print("Searching for Neighbor.... {}, {}".format(rank_node, max_search_i))
            while number_searchs < max_search_i and returned_neighbor is -1:
                # if number_searchs is 30% of max_search
                if number_searchs < max_search_i*0.3:
                    sigma = 10
                    gamma = 0
                # if number_searchs is 80% of max_search
                elif number_searchs < max_search_i*0.6:
                    sigma = 1
                    gamma = 0
                # if number_searchs is 80% of max_search
                elif number_searchs < max_search_i*0.9:
                    sigma = 0.25
                    gamma = 0
                # if number_searchs is 90% of max_search
                else:
                    sigma = 0.15
                    gamma = 1
                neighbor = nearest_leaf_node.generate_point(factual.copy(), data_catalog = self.data_catalog, sigma = sigma, gamma = gamma)
                neighb_df = pd.DataFrame([neighbor[self.mlmodel.feature_input_order]])
                self.neighb_df = neighb_df
                probs_p = self.mlmodel.predict_proba(neighb_df)
                if counter_taregt == np.argmax(probs_p):
                    returned_neighbor = neighbor
                    break
                number_searchs += 1
            if returned_neighbor is not -1:
                break
        # If no neighbor is found, return the factual
        if returned_neighbor is -1:
            #print("No neighbor was found")
            factual_ret = factual
            factual_ret[self._mlmodel.feature_input_order] = np.nan
            #print("returned")
            return factual_ret[self._mlmodel.feature_input_order]
        return returned_neighbor[self._mlmodel.feature_input_order]