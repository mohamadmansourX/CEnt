from carla import MLModelCatalog
from carla.data.catalog import OnlineCatalog
from carla.recourse_methods import GrowingSpheres
from sklearn.model_selection import GridSearchCV, train_test_split
import contextlib
from seed_env import seed_my_session
from sklearn.metrics import accuracy_score
import numpy as np
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
  def __init__(self, data, target):
    self.df = data
    self.target = target

class DataModels:
  def __init__(self, data_name, factuals_length = 50, logging_file = 'models_logs.txt'):
    self.data_name = data_name
    # Load dataset
    self.load_data_modesl(data_name=self.data_name, factuals_length = factuals_length)
    # Load models
    self.load_models(logging_file = logging_file)
    # Get data features
    self.get_data_features()

  # load a catalog dataset
  def load_data_modesl(self, data_name="adult", factuals_length = 50):
    self.dataset = OnlineCatalog(data_name)
    # Prepare Training and Test Data
    self.data_train, self.data_test = train_test_split(self.dataset.df, test_size=0.2)
    self.trainData = MyData(self.data_train.copy(), self.dataset.target)

    # load artificial neural network from catalog
    self.factuals = self.data_test.sample(factuals_length)

  # Load models by training data
  def load_models(self, logging_file = 'models_logs.txt'):
    dataset = self.dataset
    print("Loading models... --- logs will be saved to {}".format(logging_file))
    with contextlib.redirect_stdout(open(logging_file, 'w')):
      # Define models configs
      parms_training = {"ann":    {"learning_rate": 0.002, "epochs": 10, "batch_size": 128, "hidden_size": [13,4]},
                        "linear": {"learning_rate": 0.002, "epochs": 50, "batch_size": 128, "hidden_size": [13,4]},
                        "forest": {"max_depth": 2, "n_estimators": 5}}
      # Define models_zoo to store models
      self.models_zoo = {"ann": {"tensorflow": '', "pytorch": ''}, 
                    "linear": {"tensorflow": '', "pytorch": ''},
                    "forest": {"xgboost": '', "sklearn": ''}}
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
    with open(logging_file, 'a') as f:
      f.write("\n")
      f.write("Models loaded successfully!\n")
      f.write("\n")
      for model_type in self.models_zoo:
        f.write("Model type: {}\n".format(model_type))
        for framework in self.models_zoo[model_type]:
          f.write("\tFramework: {}\taccuracy: {:.3f}\n".format(framework,score_acc(self.models_zoo[model_type][framework])))
        
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
