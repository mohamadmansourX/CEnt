"""
Carla_Wrapper.ipynb
"""

# !git clone https://github.com/carla-recourse/CARLA.git


import enum
from typing import Dict, List, Tuple, Union
import pandas as pd
from carla import RecourseMethod
from carla.data.api import data, Data
from carla.models.api import MLModel
from carla.recourse_methods.autoencoder import (
    VAEDataset,
    VariationalAutoencoder,
    train_variational_autoencoder,
)
from carla.recourse_methods.processing import (
    check_counterfactuals,
    merge_default_parameters,
    reconstruct_encoding_constraints,
)
# For Descision Tree implementation
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from carla import Benchmark
from carla.recourse_methods import Dice, Face
import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import random
from sklearn.model_selection import train_test_split

class LeafNode:
    def __init__(self, conditions, label, weight):
        # Conditions is a list of tuples from the root node to the leaf node
        self.conditions = conditions
        # Label is the label of the leaf node
        self.label = label
        # Wieght Either entropy or gini
        self.weight = weight
    
    def __repr__(self):
        """
        Print the leaf node with conditions and label
        """
        return "LeafNode(label={}, weight={}, conditions={})".format(self.label, self.weight, self.conditions)

    def compare_node(self, other): #TODO misleading name
        """
        Get the distance between two leaf nodes by returning a set of conditions as follows:
        1. Initialize conditions as other conditions
        2. Remove conditions that are exactly the same with self
        3. Return the remaining conditions

        # TODO common feature
        """
        # Initialize conditions as other conditions
        conditions = other.conditions
        # Remove conditions that are common with self
        for condition in self.conditions:
            conditions = [c for c in conditions if c != condition]
        # Return the remaining conditions
        return conditions

    def check_point(self, point):
        """
        Check if the point satisfies the conditions of the leaf node
        """
        # Check if the point satisfies the conditions of the leaf node
        for condition in self.conditions:
            if not condition.check_point(point):
                return False
        return True
        
    def generate_point(self, point, min_bias=0.01, max_bias=1, force_bias = False, data_catalog = None):
      """
      Generate a point from a point
      """
      # Get Common conditions
      # TODO optimize this
      conditions_checked = []
      conditions_index = []
      # Loop over the conditions
      for c_ind, condition in enumerate(self.conditions):
            if condition.feature not in conditions_checked:
                conditions_checked.append(condition.feature)
                conditions_index.append([c_ind])
            else:
                cond_i = conditions_checked.index(condition.feature)
                conditions_index[cond_i].append(c_ind)
      #print("conditions_checked: ",conditions_checked)
      #print("conditions_index: ",conditions_index)
      for c_ind in range(len(conditions_checked)):
          if len(conditions_index[c_ind])>1:
              m_min_bias = None
              m_max_bias = None
              # Get min max threshold
              for jj in conditions_index[c_ind]:
                  # Check if its less than
                  if self.conditions[jj].threshold_sign == '<=':
                      if m_min_bias:
                        if self.conditions[jj].threshold < m_min_bias:
                          m_min_bias = self.conditions[jj].threshold
                      else:
                        m_min_bias = self.conditions[jj].threshold
                  elif self.conditions[jj].threshold_sign == '>':
                      if m_max_bias:
                        if self.conditions[jj].threshold > m_max_bias:
                          m_max_bias = self.conditions[jj].threshold
                      else:
                        m_max_bias = self.conditions[jj].threshold
              #print("Min is {} while Max is {}".format(m_min_bias,m_max_bias))
              if m_min_bias and m_max_bias:
                  bias_term_val = random.uniform(0,m_max_bias-m_min_bias)
                  #print("bias: ",bias_term_val)
                  point[conditions_checked[c_ind]] = m_max_bias - bias_term_val
              elif m_min_bias:
                  m_max_bias = max(max_bias, m_min_bias*2)
                  bias_term_val = random.uniform(0,m_max_bias-m_min_bias)
                  point[conditions_checked[c_ind]] = m_min_bias - bias_term_val
              elif m_max_bias:
                  m_min_bias = min(m_max_bias/2,min_bias)
                  # TODO add standard deviation
                  bias_term_val = random.uniform(0,m_max_bias-m_min_bias)
                  #print("Min: {}, Max: {}, Bias: {}".format(m_min_bias,
                  #                                          m_max_bias,
                  #                                          bias_term_val))
                  point[conditions_checked[c_ind]] = m_max_bias + bias_term_val
          else:
            condition = self.conditions[conditions_index[c_ind][0]]
            point = condition.sample_from_point(point, min_bias, max_bias, force_bias = force_bias)

      return point


class Condition:
    def __init__(self, feature, threshold, threshold_sign):
        # Feature is the feature name
        self.feature = feature
        # Value is the value of the feature
        self.threshold = threshold
        # <= or > since they are the only two threshold_sign in Decision Tree
        self.threshold_sign = threshold_sign
    def __repr__(self):
        return f'{self.feature} {self.threshold_sign} {self.threshold}'
    def check_point(self, point):
        """
        Check if the point satisfies the condition
        """
        # Check if the point satisfies the condition
        if self.threshold_sign == '<=':
            return point[self.feature] <= self.threshold
        else:
            return point[self.feature] > self.threshold
    def sample_from_point(self, point, min_bias=0.01, max_bias=1, force_bias = False):
        """
        Check if the point satisfies the condition
        """
        # Check if the point satisfies the condition
        if not force_bias:
            if self.check_point(point):
                return point
        sign_bias = -1 if self.threshold_sign == "<=" else 1
        new_val = self.threshold + sign_bias * random.uniform(min_bias,max_bias)
        # standard dev/10 e.g.
        #print("\t{} changed from {} to {}".format(self.feature,
        #                                        point[self.feature],
        #                                        new_val))
        point[self.feature] = new_val
        return point



class TreeLeafs:
    def __init__(self, tree, feature_input_order):
        self.tree = tree
        self.feature_input_order = feature_input_order
        self.leafs_nodes = []
        self.get_leaf_nodes(tree)

    def get_leaf_nodes(self, tree, node_id=0, conditions=[]):
        """
        This will be a recursion function that will append to leaf_nodes list, their labels and set of conditions
        If the node is a leaf node, then it will append a LeafNode object to leaf_nodes
        If the node is not a leaf node, then it will return while adding the conditions of the left and right child to the list
        """
        # If the node is a leaf node
        if tree.children_left[node_id] == -1 and tree.children_right[node_id] == -1:
            # Append the leaf node to the list
            self.leafs_nodes.append(LeafNode(conditions, np.argmax(tree.value[node_id]), tree.impurity[node_id]))
        # If the node is not a leaf node
        else:
            # Need to get the feature of the node
            feature = self.feature_input_order[tree.feature[node_id]]
            # Need to get the threshold of the node
            threshold = tree.threshold[node_id]
            # For right child if exists, threshold_sign is >
            if tree.children_right[node_id] != -1:
                conditions_right = conditions.copy()
                # Append the condition to the list
                conditions_right.append(Condition(feature, threshold, '>'))
                # Get the right child
                self.get_leaf_nodes(tree, tree.children_right[node_id], conditions_right)
            # For left child if exists, threshold_sign is <=
            if tree.children_left[node_id] != -1:
                conditions_left = conditions.copy()
                # Append the condition to the list
                conditions_left.append(Condition(feature, threshold, '<='))
                # Get the left child
                self.get_leaf_nodes(tree, tree.children_left[node_id], conditions_left)

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
          "layers": None,
          "train": True,
          "lambda_reg": 1e-6,
          "epochs": 5,
          "lr": 1e-3,
          "batch_size": 32,
      },
      "tree_params": {
          "min_entries_per_label": 100,
          "grid_search_jobs": -1,
          "min_weight_gini": 100, # set to 0.5 since here both class have same prob
          "max_search": 100,
          "grid_search": {
                "splitter": ["best"],
                "criterion": ["gini"],
                "max_depth": [5,6,7,8,9],
                "min_samples_split": [1,2,3],
                "min_samples_leaf": [1,2,3],
                "max_features": [None] #Note changing this will result in removing features that we might want to keep
          }
      }

    }

    def __init__(self, dataset:Data, mlmodel: MLModel, hyperparams: Dict, data_catalog: Dict):
        super().__init__(mlmodel)
        # Construct catalog
        self.data_catalog = data_catalog
        # Construct mlmodel
        self.mlmodel = mlmodel
        # Construct the hyperparameters
        self.hyperparams = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)
        # Construct the VAE
        self.vae = TEMP_VAE
        # self.vae = self.load_vae(dataset, self.hyperparams["vae_params"], mlmodel, mlmodel.data.name)
        # Construct the dataframe with encodings
        self.dataset = dataset.df
        self.dataset['VAE_ENCODED'] = self.get_encodeings(self.dataset)
        


    def load_vae(self, data: pd.DataFrame, vae_params: Dict, mlmodel: MLModel, data_name: str) -> VariationalAutoencoder:
        '''
        Load and train the VAE if needed
        '''
        generative_model = VariationalAutoencoder(data_name, vae_params['layers'])
        # if train is True, train the VAE
        if vae_params['train']:
            generative_model = train_variational_autoencoder(
                generative_model,
                data,
                mlmodel.feature_input_order,
                lambda_reg=vae_params["lambda_reg"],
                epochs=vae_params["epochs"],
                lr=vae_params["lr"],
                batch_size=vae_params["batch_size"],
            )
        else:
            try:
                # CHeck if the generative_model can load our data
                generative_model.load(data.shape[1] - 1)
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
        counter_factuals = check_counterfactuals(self._mlmodel, counter_factuals)
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
        clf = DecisionTreeClassifier(random_state=0)
        # Define the grid search
        grid_search = GridSearchCV(clf, self.hyperparams["tree_params"]["grid_search"], cv=5, verbose=0, refit=True, n_jobs=self.hyperparams["tree_params"]["grid_search_jobs"])
        # Fit the grid search evaluate on X_test and y_test then refit best model on the whole dataset
        grid_search.fit(train_features, target_values)
        # Return the best model
        return grid_search.best_estimator_


    def tree_based_search(self, factual):
        '''
        This method is responsible to get the counterfactual of a given targeted_encoding
        '''
        copy_data = self.dataset.copy()
        # Get distances from data to this encoding
        copy_data["distance"] = copy_data["VAE_ENCODED"].apply(lambda x: self.distance_get(x, factual["VAE_ENCODED"]))
        # Sort the dataframe by distance
        copy_data = copy_data.sort_values(by="distance")
        # Reset the index
        copy_data = copy_data.reset_index(drop=True)
        # Get the nearest neighbors of the targeted encoding
        nearest_neighbors = self.get_nearest_neighbors_thershold(copy_data, label_threshold=self.hyperparams["tree_params"]["min_entries_per_label"])
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
            max_searchs = [self.hyperparams["tree_params"]["max_search"]*0.7, self.hyperparams["tree_params"]["max_search"]*0.3]
        else:
            max_searchs = [self.hyperparams["tree_params"]["max_search"]*0.5, self.hyperparams["tree_params"]["max_search"]*0.3, self.hyperparams["tree_params"]["max_search"]*0.2]
        # map max_search to int values while rounding up to the nearest int
        max_searchs = [int(round(x)) for x in max_searchs]
        # Loop over max_search
        for rank_node, max_search_i in enumerate(max_searchs):
            number_searchs = 0
            nearest_leaf_node = leaf_nodes_with_label[rank_node]
            # print("Searching for Neighbor.... {}, {}".format(rank_node, max_search_i)))
            while number_searchs < max_search_i and returned_neighbor is -1:
                neighbor = nearest_leaf_node.generate_point(factual.copy(), data_catalog = self.data_catalog)
                if counter_taregt == np.argmax(self.mlmodel.predict_proba(pd.DataFrame([neighbor[self.mlmodel.feature_input_order]]))):
                    returned_neighbor = neighbor
                    break
                number_searchs += 1
            if returned_neighbor is -1:
                break
        # If no neighbor is found, return the factual
        if returned_neighbor is -1:
            print("No neighbor was found")
            factual_ret = factual
            factual_ret[self._mlmodel.feature_input_order] = np.nan
            #print("returned")
            return factual_ret[self._mlmodel.feature_input_order]
        return returned_neighbor[self._mlmodel.feature_input_order]

TEMP_VAE = tbtest.vae

from carla import MLModelCatalog
from carla.data.catalog import OnlineCatalog
from carla.recourse_methods import GrowingSpheres
from sklearn.model_selection import GridSearchCV, train_test_split

# load a catalog dataset
data_name = "adult"
dataset = OnlineCatalog(data_name)
data_train, data_test = train_test_split(dataset.df, test_size=0.2)

class MyData:
  def __init__(self, data, target):
    self.df = data
    self.target = target
trainData = MyData(data_train.copy(), dataset.target)

# Check our data catalog
col_n = dataset.df.columns
catalog_n = dataset.catalog
# Initialize new catalog
new_catalog_n = {'target': 'income', 'continuous': [], 'categorical': [], 'immutable': []}
# Map continuous values
for col_i in col_n:
    col = col_i.split('_')[0]
    if col == dataset.target:
        continue
    if col in catalog_n['immutable']:
        new_catalog_n['immutable'].append(col_i)
    if col in catalog_n['continuous']:
        new_catalog_n['continuous'].append(col_i)
    elif col in catalog_n['categorical']:
        new_catalog_n['categorical'].append(col_i)
    else:
        assert False, 'Column not found in catalog {}'.format(col_i)

# Assert if new_catalog_n is not same shape as catalog_n
assert len(new_catalog_n['continuous']) == len(catalog_n['continuous']), 'Continuous values not same shape'
assert len(new_catalog_n['categorical']) == len(catalog_n['categorical']), 'Categorical values not same shape'
assert len(new_catalog_n['immutable']) == len(catalog_n['immutable']), 'Immutable values not same shape'
# For each continous value get the std, mean, and min/max and plug them in the new catalog['continuous_stats']
new_catalog_n['continuous_stats'] = {}
for col_i in new_catalog_n['continuous']:
    new_catalog_n['continuous_stats'][col_i] = {}
    new_catalog_n['continuous_stats'][col_i]['std'] = data_train[col_i].std()
    new_catalog_n['continuous_stats'][col_i]['mean'] = data_train[col_i].mean()
    new_catalog_n['continuous_stats'][col_i]['min'] = data_train[col_i].min()
    new_catalog_n['continuous_stats'][col_i]['max'] = data_train[col_i].max()



# load artificial neural network from catalog
model = MLModelCatalog(dataset, 'ann')

hpr = {
      "data_name": "data_name",
      "n_search_samples": 300,
      "p_norm": 1,
      "step": 0.1,
      "max_iter": 10,
      "clamp": True,
      "binary_cat_features": True,
      "vae_params": {
          "layers": [len(model.feature_input_order), 16, 8],
          "train": True,
          "lambda_reg": 1e-6,
          "epochs": 5,
          "lr": 1e-3,
          "batch_size": 16,
      },
      "tree_params": {
          "min_entries_per_label": 300,
          "grid_search_jobs": -1,
          "min_weight_gini": 100, # set to 0.5 since here both class have same prob,
          "max_search" : 10,
          "grid_search": {
                "splitter": ["best"],
                "criterion": ["gini"],
                "max_depth": [7],
                "min_samples_split": [3],
                "min_samples_leaf": [2],
                "max_features": [None] #Note changing this will result in removing features that we might want to keep
          }
      }
    }

#julia here
tbtest = TreeBasedContrastiveExplanation(trainData, model, hpr, new_catalog_n)

factuals = data_test.sample(50)

tbtest.get_counterfactuals(factuals.copy())

# NEED TO CHECK VIOLATIONS https://github.com/carla-recourse/CARLA/blob/main/carla/evaluation/violations.py

col_n = dataset.df.columns
# print(col_n)
# >>> ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'income', 'marital-status_Non-Married', 'native-country_US', 'occupation_Other', 'race_White', 'relationship_Non-Husband', 'sex_Male', 'workclass_Private']

catalog_n = dataset.catalog
# print(catalog_n)
# >>> {'target': 'income', 'continuous': ['age', 'fnlwgt', 'education-num', 'capital-gain', 'hours-per-week', 'capital-loss'], 'categorical': ['marital-status', 'native-country', 'occupation', 'race', 'relationship', 'sex', 'workclass'], 'immutable': ['age', 'sex']}

# Need to map catalog_n column names back to col_n
# col_n are the column names after the one-hot-encoder; column-name_val1



data_train.describe()

import yaml
def load_setup() -> Dict:
    with open("experimental_setup.yaml", "r") as f:
        setup_catalog = yaml.safe_load(f)

    return setup_catalog["recourse_methods"]
setup = load_setup()



#### BENCHMARKING 

benchmark = Benchmark(model, tbtest, factuals.copy().reset_index(drop=True))
distances = benchmark.compute_distances()
benchmark.run_benchmark().head()

gs = GrowingSpheres(model)
benchmark = Benchmark(model, gs, factuals.copy().reset_index(drop=True))
distances = benchmark.compute_distances()
benchmark.run_benchmark().head()

hyperparams = setup["dice"]["hyperparams"]
dice = Dice(model, hyperparams)
benchmark = Benchmark(model, dice, factuals.copy().reset_index(drop=True))
distances = benchmark.compute_distances()
benchmark.run_benchmark().head()

hyperparams = setup["face_knn"]["hyperparams"]
face = Face(model, hyperparams)
benchmark = Benchmark(model, face, factuals.copy().reset_index(drop=True))
distances = benchmark.compute_distances()
benchmark.run_benchmark().head()
