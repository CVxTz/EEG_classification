import numpy as np
from glob import glob
import os
from sklearn.model_selection import train_test_split

base_path = "/media/ml/data_ml/EEG/deepsleepnet/data_npy"

files = glob(os.path.join(base_path, "*.npz"))
train_val, test = train_test_split(files, test_size=0.15, random_state=1337)

train, val = train_test_split(train_val, test_size=0.1, random_state=1337)

train_dict = {k: np.load(k) for k in train}
test_dict = {k: np.load(k) for k in test}
val_dict = {k: np.load(k) for k in val}



