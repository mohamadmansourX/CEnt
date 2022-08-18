"""
Tree Leaf utils
"""
import enum
from typing import Dict, List, Tuple, Union
import pandas as pd
from carla import RecourseMethod
from carla.data.api import data, Data
from carla.models.api import MLModel
from carla.recourse_methods.autoencoder import (
    VariationalAutoencoder,
    train_autoencoder,
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
from copy import deepcopy
class LeafNode:
    def __init__(self, conditions, label, weight):
        # Conditions is a list of tuples from the root node to the leaf node
        self.conditions = deepcopy(conditions)
        # Label is the label of the leaf node
        self.label = label
        # Wieght Either entropy or gini
        self.weight = weight
        # Duplicate conditions
        self.duplicate_conditions = []

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
        cond_features = [c.feature for c in conditions]
        cond_features = list(set(cond_features))
        # Return the remaining conditions
        return conditions, len(cond_features)

    def merge_conditions(self):
        """
        If there are two conditions with the same feature and threshold_sign, merge them into one condition
        """
        # Initialize conditions as other conditions
        # 
        conditions = self.conditions
        # indexes to be dropped
        indexes = []
        # Search for conditions with the same feature and threshold_sign
        for i in range(len(conditions)):
            # if i in the indexes to be dropped, skip
            if i in indexes:
                continue
            for j in range(i + 1, len(conditions)):
                # if j in the indexes to be dropped, skip
                if j in indexes:
                    continue
                if conditions[i].feature == conditions[j].feature:
                    if conditions[i].threshold_sign == conditions[j].threshold_sign:
                        # Merge the two conditions
                        if conditions[i].threshold_sign == '<=':
                            conditions[i].threshold = min(conditions[i].threshold, conditions[j].threshold)
                        else:
                            conditions[i].threshold = max(conditions[i].threshold, conditions[j].threshold)
                        # Add index to drop
                        indexes.append(j)
                    else:
                        # Add it to duplicate conditions
                        if conditions[j].feature not in self.duplicate_conditions:
                            self.duplicate_conditions.append(conditions[j].feature)
        # Drop indexes from conditions
        conditions = [c for i, c in enumerate(conditions) if i not in indexes]
        self.conditions = conditions
        self.duplicate_conditions = list(set(self.duplicate_conditions))

    def check_point(self, point):
        """
        Check if the point satisfies the conditions of the leaf node
        """
        # Check if the point satisfies the conditions of the leaf node
        for condition in self.conditions:
            if not condition.check_point(point):
                return False
        return True

    def generate_point(self, point, data_catalog = None, sigma =0.5, gamma = 0):
        """
        Generate a point from a point
        """
        # loop through the duplicate conditions
        for feature in self.duplicate_conditions:
            # get the two conditions with that feature
            conditions = [c for c in self.conditions if c.feature == feature]
            # data_catalog contains {'categorical':[],'continuous':[],'imutable':[], 'continuous_stats':[]}
            if feature in data_catalog['categorical']:
                # Assert that there shouldn't be duplicate conditions for a binary feature (categorical here are binaries)
                assert False, "There shouldn't be duplicate conditions for a binary feature"
                #TODO (general user he can't solve it)
                # Thrsh can be continous, then generate a random point between threshold and round the result
            elif feature in data_catalog['continuous']:
                # using the continuous_stats get the std and mean
                std = data_catalog['continuous_stats'][feature]['std']
                mean = data_catalog['continuous_stats'][feature]['mean']
                minn = data_catalog['continuous_stats'][feature]['min']
                maxx = data_catalog['continuous_stats'][feature]['max']
                # Using the mean, std, min and max create a bias value
                # bias values is std/10 * (max - min)
                bias = std / sigma # TOCHECK LATER
                bias = min(bias, abs(conditions[0].threshold - conditions[1].threshold))
                # Min
                if gamma == 0:
                    min_bias = 0
                else:
                    min_bias = std / gamma
                # Generate a random value between the two thresholds
                bias = random.uniform(min_bias, bias)
                # Add the bias to the threshold
                if conditions[0].threshold_sign == '<=':
                    point[feature] = conditions[0].threshold + bias
                else:
                    point[feature] = conditions[1].threshold + bias
        for condition in self.conditions:
            if condition.feature not in self.duplicate_conditions:
                if condition.feature in data_catalog['categorical']:
                    # Simply flip the value
                    # Round the threshold
                    point[condition.feature] = not point[condition.feature]
                else: # condition.feature in data_catalog['continuous']:
                    std = data_catalog['continuous_stats'][condition.feature]['std']
                    mean = data_catalog['continuous_stats'][condition.feature]['mean']
                    minn = data_catalog['continuous_stats'][condition.feature]['min']
                    maxx = data_catalog['continuous_stats'][condition.feature]['max']
                    bias = std / sigma
                    # Min
                    if gamma == 0:
                        min_bias = 0
                    else:
                        min_bias = std / gamma
                    # Generate a random value between the two thresholds
                    bias = random.uniform(min_bias, bias)
                    if condition.threshold_sign == '<=':
                        point[condition.feature] = condition.threshold + bias
                    else:
                        point[condition.feature] = condition.threshold - bias
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


class TreeLeafs:
    def __init__(self, tree, feature_input_order):
        self.tree = tree
        self.feature_input_order = feature_input_order
        self.leafs_nodes = []
        self.get_leaf_nodes(tree)
        for leaf in self.leafs_nodes:
            leaf.merge_conditions()

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
