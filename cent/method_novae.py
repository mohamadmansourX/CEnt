from tqdm.notebook import tqdm
from pynndescent import NNDescent
import enum
from typing import Dict, List, Tuple, Union
import pandas as pd
from carla import RecourseMethod
from carla.data.api import data, Data
from carla.models.api import MLModel
from cent.vae import VariationalAutoencoder
from carla.recourse_methods.processing import merge_default_parameters
from cent.TreeLeaf import TreeLeafs
# For Descision Tree implementation
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

class CEntNoVAE(RecourseMethod):
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
      "treeWarmUp": 5,
      "target_class": [0, 1],
      "binary_cat_features": True,
      "myvae_params": {
          'input_dim': 784,
          'kld_weight': 0.0025,
          'layers': [512, 128],
          'latent_dim': 32,
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
          'save_dir': './vae_model/',
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
        # No need to change the target of dataset here since VAE trained only on input features
        # self.vae = self.load_vae(dataset, self.hyperparams["myvae_params"], mlmodel, mlmodel.data.name)
        # Define feature_input
        self.feature_input_order = []
        for fin in self.mlmodel.feature_input_order:
            if fin in dataset.immutables:
                continue
            else:
                self.feature_input_order.append(fin)
        # Construct the dataframe with encodings
        self.dataset = dataset.df.copy()
        # Change dataset target to the one predicted by mlmodel
        previous_preds_was = self.dataset[self.mlmodel.data.target].copy()
        self.dataset[self.mlmodel.data.target] = self.mlmodel.predict(self.dataset).round().astype(int)
        print("Get Encodings...")
        self.dataset['VAE_ENCODED'] = self.get_encodeings(self.dataset.copy())
        ## These are added to optimize neighbor sampling for DT which used to take ~0.4 seconds and now will be 
        # NNDescent
        self.data_indexes_m = self.dataset.index
        print("Initializing the NNDescent...")
        self.set_distance_metric_initialize_nn(self.distance_metric)
        # Load Grid Parameters
        self.hyperparams["tree_params"]["grid_search"] = self.optimize_grid(self.hyperparams["tree_params"]["grid_search"], self.dataset)
        self.tree_scores = {'Train':[], 'Test':[]}

    def set_distance_metric_initialize_nn(self, distance_metric):
        # TODO : Need to make NNDescent per target class
        self.distance_metric = distance_metric
        self.nnd = NNDescent(np.array(self.dataset["VAE_ENCODED"].values.tolist()), metric=self.distance_metric,random_state=42)
        self.nnd.prepare()
        self.nnd_positive = NNDescent(np.array(self.dataset[self.dataset[self._mlmodel.data.target]==1]["VAE_ENCODED"].values.tolist()), metric=self.distance_metric,random_state=42)
        self.nnd_positive.prepare()
        self.nnd_negative = NNDescent(np.array(self.dataset[self.dataset[self._mlmodel.data.target]==0]["VAE_ENCODED"].values.tolist()), metric=self.distance_metric,random_state=42)
        self.nnd_negative.prepare()

    def set_model(self,mlmodel):
        self._mlmodel = mlmodel
        self.mlmodel = mlmodel
    def optimize_grid(self, grid_search, df):
        """
        Optimize the grid search parameters
        #@TODO: make it on chunkc of the dataframe and return the top
        #@TODO: the chunkcs in order of encodings
        """
        copy_data = df.copy()
        print("DT Warming Up on {} fits...".format(self.hyperparams['treeWarmUp']))
        # create a frequency count to count how many times a parameter was selected as best_params
        best_params_list = []
        for i in range(self.hyperparams['treeWarmUp']):
            index_factual = df.sample(1).index[0]
            factual = df.loc[index_factual]
            #index_neighbors_0 = self.nnd.query(np.array([factual["VAE_ENCODED"].tolist()]), k=self.hyperparams["tree_params"]["min_entries_per_label"]*2.5)[0][0].tolist()
            #datata_index_0 = self.data_indexes_m[index_neighbors_0].tolist()
            index_neighbors_0 = self.nnd_negative.query(np.array([factual["VAE_ENCODED"].tolist()]), k=self.hyperparams["tree_params"]["min_entries_per_label"])[0][0].tolist()
            datata_index_0 = self.data_indexes_m[index_neighbors_0].tolist()
            index_neighbors_1 = self.nnd_positive.query(np.array([factual["VAE_ENCODED"].tolist()]), k=self.hyperparams["tree_params"]["min_entries_per_label"])[0][0].tolist()
            datata_index_1 = self.data_indexes_m[index_neighbors_1].tolist()
            datata_index_0.extend(datata_index_1)
            nearest_neighbors = copy_data.loc[datata_index_0]
            # Define Decision Tree Classifier
            dec_tree = DecisionTreeClassifier(random_state=0)
            # Define Grid Search
            grid_tree = GridSearchCV(dec_tree, grid_search, cv=2, n_jobs=self.hyperparams["tree_params"]["grid_search_jobs"])
            target_values = nearest_neighbors[self._mlmodel.data.target]
            train_features = nearest_neighbors[self.feature_input_order]
            # Fit the Grid Search
            grid_tree.fit(train_features, target_values)
            best_params_list.append(grid_tree.best_params_)
        # For each key check the most common value and return that or just return random value
        best_params = {}
        best_params_listt = pd.DataFrame(best_params_list)
        for key in grid_tree.best_params_:
            best_params[key] = best_params_listt[key].value_counts().index[0]
            if key == 'min_samples_split':
                if best_params[key] >1.1:
                    best_params[key] = int(best_params[key])
        # Return the best parameters
        print(best_params)
        return best_params

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
            data_name, vae_params["layers"], mutable_list, self.hyperparams['myvae_params']
        )
        if vae_params["train"]:
            generative_model.fit(
                xtrain=data.df[mlmodel.feature_input_order]
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
        factuals[self.mlmodel.data.target] = self.mlmodel.predict(factuals).round().astype(int)
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
        # Get the encoded features
        # encoded_values = self.vae.get_encodings(input_data)
        # get encoded values without VAE
        encoded_values = input_data.values
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
        training_features = nearest_neighbors[self.feature_input_order]
        # Split the data into train and test
        train_features, test_features, target_values_train, target_values_test = train_test_split(training_features, target_values, 
                                                                                                test_size=0.1, random_state=42)
        # Create the decision tree
        clf = DecisionTreeClassifier(random_state=0 , max_depth=self.hyperparams["tree_params"]['grid_search']["max_depth"], 
                                    min_samples_split=self.hyperparams["tree_params"]['grid_search']["min_samples_split"], 
                                    min_samples_leaf=self.hyperparams["tree_params"]['grid_search']["min_samples_leaf"], 
                                    max_features=self.hyperparams["tree_params"]['grid_search']["max_features"])
        clf.fit(train_features, target_values_train)
        # Predict the test data
        predicted__test_values = clf.predict(test_features)
        predicted__train_values = clf.predict(train_features)
        # Calculate the accuracy
        self.tree_scores['Train'].append(accuracy_score(target_values_train, predicted__train_values))
        self.tree_scores['Test'].append(accuracy_score(target_values_test, predicted__test_values))
        return clf

    def tree_based_search(self, factual):
        '''
        This method is responsible to get the counterfactual of a given targeted_encoding
        '''
        copy_data = self.dataset.copy()
        #index_neighbors_0 = self.nnd.query(np.array([factual["VAE_ENCODED"].tolist()]), k=self.hyperparams["tree_params"]["min_entries_per_label"]*2.5)[0][0].tolist()
        #datata_index_0 = self.data_indexes_m[index_neighbors_0].tolist()
        index_neighbors_0 = self.nnd_negative.query(np.array([factual["VAE_ENCODED"].tolist()]), k=self.hyperparams["tree_params"]["min_entries_per_label"])[0][0].tolist()
        datata_index_0 = self.data_indexes_m[index_neighbors_0].tolist()
        index_neighbors_1 = self.nnd_positive.query(np.array([factual["VAE_ENCODED"].tolist()]), k=self.hyperparams["tree_params"]["min_entries_per_label"])[0][0].tolist()
        datata_index_1 = self.data_indexes_m[index_neighbors_1].tolist()
        datata_index_0.extend(datata_index_1)
        nearest_neighbors = copy_data.loc[datata_index_0]
        # Get the tree
        tree = self.decision_tree(nearest_neighbors)
        self.mtree = tree
        # Get the leaf nodes
        leaf_nodes = TreeLeafs(tree.tree_, self.feature_input_order).leafs_nodes.copy()
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
        if leaf_nodes[leaf_node_n_i].label != factual[self._mlmodel.data.target]:
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
        leaf_nodes_with_label = sorted(leaf_nodes_with_label, key=lambda x: leaf_node_n.compare_node(x)[1])
        # Get the counterfactual
        returned_neighbor = -1
        counter_taregt = factual[self._mlmodel.data.target]*-1 +1
        #print("Searching for Neighbor....")
        if len(leaf_nodes_with_label) == 1:
            max_searchs = [self.hyperparams["tree_params"]["max_search"]]
            #print("Distance to nearest leaf node: {}".format(leaf_node_n.compare_node(leaf_nodes_with_label[0])[1]))
        elif len(leaf_nodes_with_label) == 2:
            max_searchs = [self.hyperparams["tree_params"]["max_search"]*0.8, self.hyperparams["tree_params"]["max_search"]*0.2]
            #print("Distance to nearest leaf node: {} then {}".format(leaf_node_n.compare_node(leaf_nodes_with_label[0])[1], 
            #                                                         leaf_node_n.compare_node(leaf_nodes_with_label[1])[1]))
        else:
            if leaf_node_n.compare_node(leaf_nodes_with_label[2])[1] - leaf_node_n.compare_node(leaf_nodes_with_label[0])[1] < 2:
                max_searchs = [self.hyperparams["tree_params"]["max_search"]*0.6, self.hyperparams["tree_params"]["max_search"]*0.2,
                               self.hyperparams["tree_params"]["max_search"]*0.2]
                #print("Distance to nearest leaf node: {} then {} then {}".format(leaf_node_n.compare_node(leaf_nodes_with_label[0])[1], 
                #                                                        leaf_node_n.compare_node(leaf_nodes_with_label[1])[1], 
                #                                                        leaf_node_n.compare_node(leaf_nodes_with_label[2])[1]))
            else:
                max_searchs = [self.hyperparams["tree_params"]["max_search"]*0.8, self.hyperparams["tree_params"]["max_search"]*0.2]
                #print("Distance to nearest leaf node: {} then {}".format(leaf_node_n.compare_node(leaf_nodes_with_label[0])[1], 
                #                                                        leaf_node_n.compare_node(leaf_nodes_with_label[1])[1]))

        # map max_search to int values while rounding up to the nearest int
        max_searchs = [int(round(x)) for x in max_searchs]
        self.found_node = None
        # Loop over max_search
        for rank_node, max_search_i in enumerate(max_searchs):
            number_searchs = 0
            nearest_leaf_node = leaf_nodes_with_label[rank_node]
            #print("Searching for Neighbor.... {}, {}".format(rank_node, max_search_i))
            while number_searchs < max_search_i and returned_neighbor is -1:
                if number_searchs < max_search_i*0.2:
                    sigma = 20
                    gamma = 0
                elif number_searchs < max_search_i*0.4:
                    sigma = 5
                    gamma = 0
                elif number_searchs < max_search_i*0.8:
                    sigma = 1
                    gamma = 0
                else:
                    sigma = 0.15
                    gamma = 0
                neighbor = nearest_leaf_node.generate_point(factual.copy(), data_catalog = self.data_catalog, sigma = sigma, gamma = gamma)
                neighb_df = pd.DataFrame([neighbor[self.mlmodel.feature_input_order]])
                self.neighb_df = neighb_df
                probs_p = self.mlmodel.predict_proba(neighb_df)
                if counter_taregt == np.argmax(probs_p):
                    returned_neighbor = neighbor
                    self.found_node = nearest_leaf_node
                    break
                number_searchs += 1
            if returned_neighbor is not -1:
                break
        # If no neighbor is found, return the factual
        if returned_neighbor is -1:
            #print("No neighbor was found")
            factual_ret = factual
            factual_ret[self._mlmodel.feature_input_order] = np.nan
            return factual_ret[self._mlmodel.feature_input_order]
        return returned_neighbor[self._mlmodel.feature_input_order]
