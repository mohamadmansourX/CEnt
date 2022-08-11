import os
from carla import MLModelCatalog
from carla.data.catalog import OnlineCatalog
from carla.recourse_methods import GrowingSpheres
from sklearn.model_selection import GridSearchCV, train_test_split
import contextlib
from seed_env import seed_my_session
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import torchvision
from carla.data.api import data, Data
from tqdm.auto import tqdm

# Seed the environment
seed_my_session()

def score_acc(model):
    dataset = model.data
    data = dataset.df
    input_cols = model.feature_input_order
    target_column = dataset.target
    X = data[input_cols]
    y = data[target_column]
    y_pred = model.predict(X)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    accuracy = accuracy_score(y, y_pred)
    return accuracy

class MyData:
  def __init__(self, data, target, immutables):
    self.df = data
    self.target = target
    self.immutables = immutables

def load_mnist_fashionmnist(data_name):
    if data_name == 'fashionmnist':
        data = torchvision.datasets.FashionMNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ]))
    elif data_name == 'mnist':
        print(data_name)
        data = torchvision.datasets.MNIST('/files/', train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ]))
    else:
        raise ValueError('data_name must be either "mnist" or "fashionmnist"')
    # Load data to pandas DataFrame
    df_values = []
    for i in tqdm(range(len(data))):

        # Get the image and the label
        image = data[i][0]
        # Reshape the image from (28,28) to (784)
        image = image.view(28*28)
        # Get the label
        label = data[i][1]
        # Create a list where the first element is 0x0 pixel, second element is the 0x1 pixel, etc. and last element is the label
        # Get the pixels liste form image
        pixels = image.tolist()
        # Create a list where the first element is 0x0 pixel, second element is the 0x1 pixel, etc. and last element is the label
        pixels = pixels + [label]
        # Append the pixels list to the df_values list
        df_values.append(pixels)

    # Create a pandas DataFrame from the df_values list
    # The columns names will be e.g. ["0x0","0x1",...."label"]
    columns=[]
    for i in range(28):
        for j in range(28):
            columns.append(str(i)+"x"+str(j))
    columns.append("label")
    df_data = pd.DataFrame(df_values, columns=columns)
    return df_data

class MnistData(Data):
    def __init__(self, data_name, labels_needed, n= 28):
        # the dataset could be loaded in the constructor
        dataset = load_mnist_fashionmnist(data_name)
        dataset = dataset[dataset['label'].isin(labels_needed)]
        dataset['label'] = dataset['label'].map({labels_needed[0]: 0, labels_needed[1]: 1})
        self._dataset = dataset
        for coli in self._dataset.columns:
            if coli == 'label':
                continue
            #self._dataset[coli] = self._dataset[coli] / 255.0
        self._identity_encoding = True
        self.df_train, self.df_test = train_test_split(self._dataset, test_size=0.2)
        self.continuous = []
        # self.categorical = ["pixel_"+str(i)+"_"+str(j) for i in range(n) for j in range(n)]
        self.categorical = [str(i)+"x"+str(j) for i in range(n) for j in range(n)]
        self.immutables = []
        self.target = 'label'
        self.name = 'mnist'
        self.df = self._dataset #.drop(columns=self.target)
        self.catalog = {'categorical': self.categorical,
                      'continuous': self.continuous,
                      'immutable': self.immutables,
                      'target': self.target}
    def categorical(self):
        # this property contains a list of all categorical features
        return self.categorical
    def continuous(self):
        # this property contains a list of all continuous features
        return self.continuous

    def immutables(self):
        # this property contains a list of features which should not be changed by the recourse method
        return self.immutables

    def target(self):
        # this property contains the feature name of the target column
        return self.target

    def raw(self):
        # this property contains the not encoded and not normalized, raw dataset
        return self._dataset

    def df(self):
        return self._dataset
    def df_test(self):
        return self.df_test 

    def df_train(self):
        return self.df_train 
        
    def inverse_transform(self):
        return self._dataset 

    def transform(self):
        return self._dataset

class DataImagesModels:
  def __init__(self, data_name, factuals_length = 50, out_dir = '', labels_needed = [3,6]):
    logging_file = os.path.join(out_dir, 'models_logs.txt')
    self.models_metrics_file = os.path.join(out_dir, 'model_zoo_metrics.csv')
    self.data_name = data_name
    # Load dataset
    self.load_data_modesl(data_name=self.data_name, factuals_length = factuals_length, labels_needed = labels_needed)
    # Load models
    self.load_models(logging_file = logging_file)
    # Get data features
    self.get_data_features()

  # load a catalog dataset
  def load_data_modesl(self, data_name="", factuals_length = 50, labels_needed = [3,6]):
    self.dataset = MnistData(data_name=data_name, labels_needed = labels_needed)
    # Prepare Training and Test Data
    # test_size is the percentages of factuals to be used for testing
    factuals_length_percentage = factuals_length/self.dataset.df.shape[0] * 3
    self.data_train, self.data_test = train_test_split(self.dataset.df, test_size=factuals_length_percentage)
    # Fill immutables with list of True the same length of the dataframe columns
    self.immutables = []
    self.trainData = MyData(self.data_train.copy(), self.dataset.target, self.immutables)

    # load artificial neural network from catalog
    self.factuals = self.data_test.sample(factuals_length)

  # Load models by training data
  def load_models(self, logging_file = 'models_logs.txt'):
    dataset = self.dataset
    print("Loading models... --- logs will be saved to {}".format(logging_file))
    with contextlib.redirect_stdout(open(logging_file, 'w')):
      # Define models configs
      parms_training = {"ann":    {"learning_rate": 0.002, "epochs": 4, "batch_size": 64, "hidden_size": [13,4]},
                        "linear": {"learning_rate": 0.002, "epochs": 4, "batch_size": 64, "hidden_size": [13,4]}}
      #                  "forest": {"max_depth": 10, "n_estimators": 5}}
      # Define models_zoo to store models
      self.models_zoo = {"ann": {"tensorflow": '', "pytorch": ''}, 
                    "linear": {"tensorflow": '', "pytorch": ''}}
      #              "forest": {"xgboost": '', "sklearn": ''}}
      # Start filling models_zoo
      for model_type in self.models_zoo:
        for framework in self.models_zoo[model_type]:
          # Load model from catalog
          model = MLModelCatalog(
                      dataset,
                      model_type=model_type,
                      load_online=False,
                      backend=framework)
          # Train model
          model.train(**parms_training[model_type])
          
          # Save model
          self.models_zoo[model_type][framework] = model
    # Save model metrics
    frameworks = []
    model_types = []
    accuracies = []
    for model_type in self.models_zoo:
      for framework in self.models_zoo[model_type]:
        frameworks.append(framework)
        model_types.append(model_type)
        accuracies.append(score_acc(self.models_zoo[model_type][framework]))
    df = pd.DataFrame({"framework": frameworks, "model_type": model_types, "accuracy": accuracies})
    df.to_csv(self.models_metrics_file , index=False)
        
  def get_data_features(self):
    # Check our data catalog
    col_n = self.dataset.df.columns
    catalog_n = self.dataset.catalog
    # Initialize new catalog
    self.new_catalog_n = {'target': 'income', 'continuous': [], 'categorical': [], 'immutable': []}
    # Map continuous values
    for col_i in col_n:
        # Assuming one hot encoder will map new columns after '_'
        col = col_i.split('_')[0]
        if col == self.dataset.target:
            continue
        if col in catalog_n['immutable']:
            self.new_catalog_n['immutable'].append(col_i)
        if col in catalog_n['continuous']:
            self.new_catalog_n['continuous'].append(col_i)
        elif col in catalog_n['categorical']:
            self.new_catalog_n['categorical'].append(col_i)
        else:
            # Check if it is contained somewhere
            col = col_i
            if self.dataset.target in col:
                continue
            for im_ctn in catalog_n['immutable']:
                if im_ctn in col:
                    self.new_catalog_n['immutable'].append(col_i)
                    break
            not_continuous_flag = True
            not_categorical_flag = True
            for im_ctn in catalog_n['continuous']:
                if im_ctn in col:
                    self.new_catalog_n['continuous'].append(col_i)
                    not_continuous_flag = False
                    break
            if not_continuous_flag:
                for im_ctn in catalog_n['categorical']:
                    if im_ctn in col:
                        self.new_catalog_n['categorical'].append(col_i)
                        not_categorical_flag = False
                        break
            if not_categorical_flag and not_continuous_flag:
                assert False, 'Column not found in catalog {}, original {}\n{}'.format(col_i, col, catalog_n)

    # Assert if self.new_catalog_n is not same shape as catalog_n
    assert len(self.new_catalog_n['continuous']) == len(catalog_n['continuous']), 'Continuous values not same shape'
    assert len(self.new_catalog_n['categorical']) == len(catalog_n['categorical']), 'Categorical values not same shape'
    assert len(self.new_catalog_n['immutable']) == len(catalog_n['immutable']), 'Immutable values not same shape'
    # For each continous value get the std, mean, and min/max and plug them in the new catalog['continuous_stats']
    self.new_catalog_n['continuous_stats'] = {}
    for col_i in self.new_catalog_n['continuous']:
        self.new_catalog_n['continuous_stats'][col_i] = {}
        self.new_catalog_n['continuous_stats'][col_i]['std'] = self.data_train[col_i].std()
        self.new_catalog_n['continuous_stats'][col_i]['mean'] = self.data_train[col_i].mean()
        self.new_catalog_n['continuous_stats'][col_i]['min'] = self.data_train[col_i].min()
        self.new_catalog_n['continuous_stats'][col_i]['max'] = self.data_train[col_i].max()
