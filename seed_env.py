import contextlib
import torch
import tensorflow as tf
import numpy as np
import random
import os
import torch
import tensorflow as tf
import numpy as np
import random
import sklearn
import xgboost
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
def seed_my_session(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  tf.compat.v1.set_random_seed(42)
  sklearn.random_state = seed
  os.environ['PYTHONHASHSEED'] = str(seed)