import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings

import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)

import torch
torch.cuda.is_available = lambda : False
import yaml
from seed_env import seed_my_session
from typing import Dict, List
from cote.data_specific import DataModels
from carla import Benchmark
import pandas as pd
from carla.recourse_methods import (
    CCHVAE,
    CEM,
    CRUD,
    FOCUS,
    ActionableRecourse,
    CausalRecourse,
    Clue,
    Dice,
    Face,
    FeatureTweak,
    GrowingSpheres,
    Revise,
    Wachter,
)
from carla.recourse_methods.catalog.causal_recourse import constraints, samplers
import carla.evaluation.catalog as evaluation_catalog
from cote.TreeResource import TreeBasedContrastiveExplanation
# Filter warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

seed_my_session()
def load_setup() -> Dict:
    with open("experimental_setup.yaml", "r") as f:
        setup_catalog = yaml.safe_load(f)
    return setup_catalog["recourse_methods"]

def get_resource_supported_backend(recourse_method, supported_backend_dict):
    '''
    Search in supported_backend for the recourse_method to know what backend is supported
    '''
    # supported_backend contains 'pytorch', 'tensorflow', 'xgboost', 'sklearn'
    # Check what backend contains the recourse_method
    suuported_backs = []
    for backend in supported_backend_dict:
        if recourse_method in supported_backend_dict[backend]:
            suuported_backs.append(backend)
    # If tensorflow and pytorch are in the list, keep only tensorflow
    if "tensorflow" in suuported_backs and "pytorch" in suuported_backs:
        # Remove pytorch from the list
        suuported_backs.remove("pytorch")
    # Similarly for xgboost and sklearn, keep xgboost
    if "xgboost" in suuported_backs and "sklearn" in suuported_backs:
        suuported_backs.remove("sklearn")
    #TODO: Keep both, but temp return only first
    return suuported_backs[0]
    return suuported_backs

def intialialize_recourse_method(method, hyperparams, mlmodel, data_models):
    # TODO restrict data to training only
    if method == "cchvae":
        hyperparams["data_name"] = data_name
        hyperparams["vae_params"]["layers"] = [
            len(mlmodel.feature_input_order)
        ] + hyperparams["vae_params"]["layers"]
        return CCHVAE(mlmodel, hyperparams)
    elif "cem" in method:
        hyperparams["data_name"] = data_name
        raise ValueError("Session Methods not supported yet")
        #return CEM(sess, mlmodel, hyperparams)
    elif method == "clue":
        hyperparams["data_name"] = data_name
        return Clue(mlmodel.data, mlmodel, hyperparams)
    elif method == "cruds":
        hyperparams["data_name"] = data_name
        # variable input layer dimension is first time here available
        hyperparams["vae_params"]["layers"] = [
            len(mlmodel.feature_input_order)
        ] + hyperparams["vae_params"]["layers"]
        return CRUD(mlmodel, hyperparams)
    elif method == "dice":
        return Dice(mlmodel, hyperparams)
    elif "face" in method:
        return Face(mlmodel, hyperparams)
    elif method == "growing_spheres":
        return GrowingSpheres(mlmodel)
    elif method == "revise":
        hyperparams["data_name"] = data_name
        # variable input layer dimension is first time here available
        hyperparams["vae_params"]["layers"] = [
            len(mlmodel.feature_input_order)
        ] + hyperparams["vae_params"]["layers"]
        return Revise(mlmodel, mlmodel.data, hyperparams)
    elif "wachter" in method:
        return Wachter(mlmodel, hyperparams)
    elif "causal_recourse" in method:
        hyperparams['constraint_handle'] = constraints.point_constraint
        hyperparams["sampler_handle"]= samplers.sample_true_m0
        return CausalRecourse(mlmodel, hyperparams)
    elif "focus" in method:
        hyperparams = {'optimizer': 'adam', 'lr': 0.001, 'n_class': 2, 'n_iter': 1000, 'sigma': 1.0, 'temperature': 1.0, 'distance_weight': 0.01, 'distance_func': 'l1'}
        return FOCUS(mlmodel, hyperparams)
    elif "cote" in method:
        hpr = {"data_name": "data_name","n_search_samples": 300,"p_norm": 1,"step": 0.1,"max_iter": 10,"clamp": True,
                "binary_cat_features": True,
                "vae_params": {
                    "layers": [len(mlmodel.feature_input_order), 20, 10, 7],"train": True,"lambda_reg": 1e-6,
                    "epochs": 1,"lr": 1e-3,"batch_size": 64,},
                "tree_params": {
                    "min_entries_per_label": int(data_models.trainData.df.shape[0]*0.04),
                    "grid_search_jobs": -1,
                    "min_weight_gini": 100, # set to 0.5 since here both class have same prob,
                    "max_search" : 9,
                    "grid_search": {"cv": 1,"splitter": ["best"],"criterion": ["gini"],"max_depth": [3,4,5,6,7,8,9,10],
                                    "min_samples_split": [1.0,2,3],"min_samples_leaf": [1,2,3],"max_features": [None] # Changing this --> removing features
                                    }
                }
          }
        return TreeBasedContrastiveExplanation(data_models.trainData, mlmodel, hpr, data_catalog= data_models.new_catalog_n)

    else:
        raise ValueError("Recourse method not known  {}".format(method))


setup_catalog = load_setup()

# data_names = ['adult', 'compas', 'give_me_some_credit', 'heloc']
supported_backend_dict = {'pytorch': ["cchvae", "clue", "cruds", "dice", "face", 'growing_spheres',"revise" 'wachter', 
                                    'causal_recourse','actionable_recourse'],
                        'tensorflow': ['cem', 'dice', 'face', 'growing_spheres', 'causal_recourse','actionable_recourse','cote'],
                        'sklearn': ['feature_tweak','focus'],
                        'xgboost': ['feature_tweak','focus']}


# VAE distance in benchmarking
# Github migration
# clue, dice, face, growing_spheres, [focus, cem,] crude, wama tayasar
# VAE according to data columns


# MNIST last edits

# Hyperparameters tweaking (less important)


FACTUAL_NUMBER = 2

data_names = ['adult','compas', 'give_me_some_credit', 'heloc']

recourse_methods = ['cote','clue','causal_recourse','dice','focus','cchvae','cruds','actionable_recourse',
                    'cem','growing_spheres','revisewachter','face','feature_tweak']

NOTWORKING = [] # ['causal_recourse','focus'] # NOTWORKING
TESTEDSUCCESSFULLY = ['clue','dice','cote','cchvae'] # ALREADY TESTED

recourse_methods = ['growing_spheres', 'cote']#, 'causal_recourse']

out_models = 'output/models/'
out_csvs ='output/csvs/'

# Create output directories
if not os.path.exists(out_models):
    os.makedirs(out_models)
if not os.path.exists(out_csvs):
    os.makedirs(out_csvs)

test_checks = {'Resource_Method':[], 'Success_Boolean': [], 'data_name':[],'Details':[]}
check_csv = 'output/checks.csv'

print(recourse_methods)
# Loop over datasets
for data_name in data_names:
    print('Starting experiment for dataset {}'.format(data_name))
    # Load dataset and necessary models
    data_models = DataModels(data_name = data_name,
                             factuals_length = FACTUAL_NUMBER,
                             logging_file = os.path.join(out_models,data_name+'_models_logs.txt'))
    # Define a dict to store results
    metrics_scores = []
    # Define csv file to store results
    csv_file = os.path.join(out_csvs, data_name+'_results.csv')
    # Loop over recourse methods
    for recourse_method in recourse_methods:
        print('----------------------------------------\nStarting experiment for recourse method {}\n\n'.format(recourse_method))
        try:
            if recourse_method in NOTWORKING:
                print('Skipping {} as its in the NOTWORKING list'.format(recourse_method))
                continue
            # Check supported backend
            supported_backend = get_resource_supported_backend(recourse_method, supported_backend_dict)
            if supported_backend in ['tensorflow', 'pytorch']:
                supported_types = ['linear', 'ann']
            else:
                supported_types = ['forest']
            # Initialize resource method
            # create model using first supported backend and supported type just to intialize the model
            model_temp = data_models.models_zoo[supported_types[0]][supported_backend]

            if recourse_method in setup_catalog:
                if 'hyperparams' in setup_catalog[recourse_method]:
                    hyperpars = setup_catalog[recourse_method]['hyperparams']
            else:
                hyperpars = {}

            rcmethod = intialialize_recourse_method(recourse_method, hyperpars, model_temp, data_models)
            
            # Benchmark resource method
            # Loop over supported types
            for supported_type in supported_types:
                # Load model
                model = data_models.models_zoo[supported_type][supported_backend]
                # Benchmark recourse method
                benchmark = Benchmark(model, rcmethod,  data_models.factuals.copy().reset_index(drop=True))
                # Define metrics
                measures = [
                    evaluation_catalog.YNN(benchmark.mlmodel, {"y": 5, "cf_label": 1}),
                    evaluation_catalog.Distance(benchmark.mlmodel),
                    evaluation_catalog.SuccessRate(),
                    evaluation_catalog.Redundancy(benchmark.mlmodel, {"cf_label": 1}),
                    evaluation_catalog.ConstraintViolation(benchmark.mlmodel),
                    evaluation_catalog.AvgTime({"time": benchmark.timer}),
                ]
                # Run the benchmark and return the mean
                resource_bench = benchmark.run_benchmark(measures = measures)
                # resource_bench.to_csv('resource_bench_dataname_{}_method_{}.csv'.format(data_name, recourse_method))
                resource_bench = resource_bench.mean()
                # Fill the model type and backend into the metrics_scores dict
                resource_bench['model_type'] = supported_type
                resource_bench['backend'] = supported_backend
                resource_bench['recourse_method'] = recourse_method
                # Append to metrics_scores
                metrics_scores.append(resource_bench)
                # Load to pandas dataframe
                metrics_scores_df = pd.DataFrame(metrics_scores)
                # Write to csv file
                metrics_scores_df.to_csv(csv_file, index=False)
                # Save method
                test_checks['Resource_Method'].append(recourse_method)
                test_checks['Success_Boolean'].append('success')
                test_checks['data_name'].append(data_name)
                test_checks['Details'].append('success')
                # Load test checks to pandas dataframe
                test_checks_df = pd.DataFrame(test_checks)
                # Write to csv file
                test_checks_df.to_csv(check_csv, index=False)
        except Exception as e:
            print('Exception for {}'.format(recourse_method))
            print(e)
            test_checks['Resource_Method'].append(recourse_method)
            test_checks['Success_Boolean'].append('failed')
            test_checks['Details'].append(str(e))
            test_checks['data_name'].append(data_name)
            # Load test checks to pandas dataframe
            test_checks_df = pd.DataFrame(test_checks)
            # Write to csv file
            test_checks_df.to_csv(check_csv, index=False)