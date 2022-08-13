from copy import deepcopy
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings

import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")
from carla.data.api import data
import numpy as np
import torch
#torch.cuda.is_available = lambda : False
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
from vae_benchmark import VAEBenchmark

seed_my_session()
def load_setup() -> Dict:
    with open("experimental_setup.yaml", "r") as f:
        setup_catalog = yaml.safe_load(f)
    return setup_catalog["recourse_methods"]

def print_conf(conf, d=4, d_iter=5):
    for k, v in conf.items():
        if isinstance(v, dict):
            print("{}{} : ".format(d * " ", str(k)))
            print_conf(v, d + d_iter)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            print("{}{} : ".format(d * " ", str(k)))
            for value in v:
                print_conf(value, d + d_iter)
        else:
            print("{}{} : {}".format(d * " ", k, v))
        
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
        return CausalRecourse(mlmodel, hyperparams)
    elif "focus" in method:
        hyperparams = {'optimizer': 'adam', 'lr': 0.001, 'n_class': 2, 'n_iter': 1000, 'sigma': 1.0, 'temperature': 1.0, 'distance_weight': 0.01, 'distance_func': 'l1'}
        return FOCUS(mlmodel, hyperparams)
    elif "feature_tweak" in method:
        return FOCUS(mlmodel, hyperparams)
    elif "cote" in method:
        min_entries_per_label = int(data_models.trainData.df.shape[0]*0.01)
        MIN_ENTRIES_PER_LABEL_THRESH = 500
        if min_entries_per_label<MIN_ENTRIES_PER_LABEL_THRESH:
            print('min_entries_per_label is too small {}, setting it to {} '.format(min_entries_per_label,MIN_ENTRIES_PER_LABEL_THRESH))
            min_entries_per_label = MIN_ENTRIES_PER_LABEL_THRESH
        hpr = {"data_name": "data_name","n_search_samples": 300,
                "p_norm": 1,"step": 0.1,"max_iter": 10,"clamp": True,
                "treeWarmUp": 5,
                "binary_cat_features": True,
                "myvae_params": {
                    'input_dim': len(mlmodel.feature_input_order),
                    'kld_weight': 0.00025,
                    'layers': layers,
                    'latent_dim': latent_dim,
                    'hidden_activation': 'relu',
                    'dropout': 0.2,
                    'batch_norm': True,
                    'batch_size': 512,
                    'epochs': 1,
                    'learning_rate': 0.001,
                    'weight_decay': 0.000001,
                    'cuda': False,
                    'verbose': True,
                    'train': True,
                    'save_dir': './vae_model/',
                },
                "tree_params": {
                    "min_entries_per_label": min_entries_per_label,
                    "grid_search_jobs": -1,
                    "min_weight_gini": 100, # set to 0.5 since here both class have same prob,
                    "max_search" : 10,
                    "grid_search": {"cv": 1,"splitter": ["best"],"criterion": ["gini"],"max_depth": [3,4,5,6,7,8,9,10],
                                    "min_samples_split": [1.0,2,3],"min_samples_leaf": [1,2,3],
                                    "max_features": [0.4, 0.6, 0.8],
                                    }
                }
          }
        print_conf(hpr)
        return TreeBasedContrastiveExplanation(deepcopy(data_models.trainData), mlmodel, hpr, data_catalog= data_models.new_catalog_n)

    else:
        raise ValueError("Recourse method not known  {}".format(method))


setup_catalog = load_setup()

# data_names = ['adult', 'compas', 'give_me_some_credit', 'heloc']
supported_backend_dict = {'pytorch': ["cchvae", "clue", "cruds", "dice", "face", 'growing_spheres',"revise" 'wachter', 
                                    'causal_recourse','actionable_recourse'],
                        'tensorflow': ['cem', 'dice', 'face', 'growing_spheres', 'causal_recourse','actionable_recourse','cote'],
                        'sklearn': ['feature_tweak','focus'],
                        'xgboost': ['feature_tweak','focus']}


# VAE distance in benchmarking                                                  DONE
# Github migration                                                              DONE
# clue, dice, face, growing_spheres, [focus, cem,] crude, wama tayasar          ~DONE
# VAE latent representation layer size according to data columns                ~DONE
# VAE constraint
# VAE encodings distance in benchmarking                                        DONE
# Implement our working version of VAE (tested on MNIST with 2 neurons)         DONE
# Fix Best Metric in our version of VAE (load the best instead of using latest) DONE
# Save Benchmark results per row, factuals, counterfactuals                     DONE
# Save Tree results per row (Some time is added for inf)                        DONE
#       For (3124, 15) nearest neighbors, we have the following timings: 
#               1. DTree fitting per row: 11.6 ms ± 1.72 ms per run
#               2. Scoring (Inference+Score): 3.9 ms ± 126 µs per run



FACTUAL_NUMBER = 50

data_names = ['adult','compas', 'give_me_some_credit', 'heloc']

recourse_methods = ['cote','dice','growing_spheres','clue','causal_recourse',
                    'cchvae','cruds','focus','actionable_recourse',
                    'cem','revisewachter','face','feature_tweak']

NOTWORKING = [] # ['causal_recourse','focus'] # NOTWORKING
TESTEDSUCCESSFULLY = ['clue','dice','cote','cchvae'] # ALREADY TESTED


data_names = ['adult','heloc']

recourse_methods = ['cote','dice','growing_spheres']


# Define Output Directory
OUT_DIR = "./outputs/"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

print(recourse_methods)

# Loop over datasets
for data_name in data_names:

    print('######################################################################')
    print('Starting experiment for dataset {}'.format(data_name))
    print('######################################################################\n')

    OUT_DIR_DATA = os.path.join(OUT_DIR, data_name)
    if not os.path.exists(OUT_DIR_DATA):
        os.makedirs(OUT_DIR_DATA)
    
    OUT_DIR_DATA_BENCH_CSVS = os.path.join(OUT_DIR_DATA, 'bench_csvs')
    if not os.path.exists(OUT_DIR_DATA_BENCH_CSVS):
        os.makedirs(OUT_DIR_DATA_BENCH_CSVS)
    
    # Load dataset and necessary models
    data_models = DataModels(data_name = data_name,
                             factuals_length = FACTUAL_NUMBER,
                             out_dir = OUT_DIR_DATA)
    
    # Load VAE
    print("Starting VAE for benchmarking")
    # Get an ann tensorflow model as temp just to get some hyperparams
    temp_model = data_models.models_zoo['ann']['tensorflow']
    
    if len(temp_model.feature_input_order) > 500:
        layers = [500, 250]
        latent_dim = 32
    elif len(temp_model.feature_input_order) > 100:
        layers = [100, 50]
        latent_dim = 24
    elif len(temp_model.feature_input_order) > 50:
        layers = [50, 25]
        latent_dim = 16
    elif len(temp_model.feature_input_order) > 20:
        layers = [25, 16]
        latent_dim = 12
    elif len(temp_model.feature_input_order) > 10:
        layers = [25]
        latent_dim = 8
    else:
        layers = [16]
        latent_dim = 7
    xxmutables = []
    for i in range(len(temp_model.feature_input_order)):
        xxmutables.append(True)
    xxmutables = np.array(xxmutables)
    vae_parms = { 
        "myvae_params": {
            'input_dim': len(temp_model.feature_input_order),
            'kld_weight': 0.00025,
            'layers': layers,
            'latent_dim': latent_dim,
            'hidden_activation': 'relu',
            'dropout': 0.2,
            'batch_norm': True,
            'batch_size': 512,
            'epochs': 1,
            'learning_rate': 0.001,
            'weight_decay': 0.000001,
            'cuda': False,
            'verbose': True,
            'train': True,
            'save_dir': './vae_model/',
        }
    }
    print_conf(vae_parms)
    vae_bench = VAEBenchmark(temp_model, vae_parms)
    vae_bench.vae.plot_loss(plot_flag = False, save_path = os.path.join(OUT_DIR_DATA, 'loss_plot.png'))

    # Define a dict to store results
    metrics_scores = []
    # Define csv file to store results
    csv_file = os.path.join(OUT_DIR_DATA, 'benchmark_results.csv')

    # Define Checkers
    test_checks = {'Resource_Method':[], 'Success_Boolean': [], 'model_type':[],'Details':[]}
    check_csv = os.path.join(OUT_DIR_DATA, 'checks.csv')
    # Loop over recourse methods
    for recourse_method in recourse_methods:
        # Check supported backend
        supported_backend = get_resource_supported_backend(recourse_method, supported_backend_dict)
        if supported_backend in ['tensorflow', 'pytorch']:
            supported_types = ['linear', 'ann']
        else:
            supported_types = ['forest']
        print('----------------------------------------\nStarting experiment for recourse method {} in {}\n\n'.format(recourse_method,supported_types))
        
        # Benchmark resource method
        # Loop over supported types
        for supported_type in supported_types:
            if True:
                # Initialize resource method
                # create model using first supported backend and supported type just to intialize the model
                model_temp = data_models.models_zoo[supported_type][supported_backend]

                if recourse_method in setup_catalog:
                    if 'hyperparams' in setup_catalog[recourse_method]:
                        hyperpars = setup_catalog[recourse_method]['hyperparams']
                else:
                    hyperpars = {}

                rcmethod = intialialize_recourse_method(recourse_method, hyperpars, model_temp, data_models)
                # Load model
                model = data_models.models_zoo[supported_type][supported_backend]
                # Benchmark recourse method
                benchmark = Benchmark(model, rcmethod,  data_models.factuals[supported_type].copy().reset_index(drop=True))
                # Define metrics
                measures = [
                    evaluation_catalog.YNN(benchmark.mlmodel, {"y": 5, "cf_label": 1}),
                    evaluation_catalog.Distance(benchmark.mlmodel),
                    evaluation_catalog.SuccessRate(),
                    evaluation_catalog.Redundancy(benchmark.mlmodel, {"cf_label": 1}),
                    evaluation_catalog.ConstraintViolation(benchmark.mlmodel),
                    evaluation_catalog.AvgTime({"time": benchmark.timer}),
                    vae_bench
                ]
                # Run the benchmark and return the mean
                resource_bench = benchmark.run_benchmark(measures = measures)
                bench_csv = os.path.join(OUT_DIR_DATA_BENCH_CSVS, '{}_{}_{}_bench.csv'.format(recourse_method, supported_backend, supported_type))
                bench_csv_factuals = os.path.join(OUT_DIR_DATA_BENCH_CSVS, '{}_{}_{}_factuals.csv'.format(recourse_method, supported_backend, supported_type))
                bench_csv_counterfactuals = os.path.join(OUT_DIR_DATA_BENCH_CSVS, '{}_{}_{}_counterfactuals.csv'.format(recourse_method, supported_backend, supported_type))
                benchmark._factuals.to_csv(bench_csv_factuals, index=False)
                benchmark._counterfactuals.to_csv(bench_csv_counterfactuals, index=False)
                resource_bench.to_csv(bench_csv, index=False)
                if recourse_method == 'cote':
                    bench_csv_tree_scores = os.path.join(OUT_DIR_DATA_BENCH_CSVS, '{}_{}_{}_DTScores.csv'.format(recourse_method, supported_backend, supported_type))
                    pd.DataFrame(rcmethod.tree_scores).to_csv(bench_csv_tree_scores, index=False)
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
                test_checks['model_type'].append(supported_type)
                test_checks['Details'].append('success')
                # Load test checks to pandas dataframe
                test_checks_df = pd.DataFrame(test_checks)
                # Write to csv file
                test_checks_df.to_csv(check_csv, index=False)
            else:#except Exception as e:
                print('Exception for {}'.format(recourse_method))
                print(e)
                test_checks['Resource_Method'].append(recourse_method)
                test_checks['Success_Boolean'].append('failed')
                test_checks['Details'].append(str(e))
                test_checks['model_type'].append(supported_type)
                # Load test checks to pandas dataframe
                test_checks_df = pd.DataFrame(test_checks)
                # Write to csv file
                test_checks_df.to_csv(check_csv, index=False)
print("\n\nFINISHED BENCHMARKING!")
