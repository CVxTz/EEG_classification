import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import collections
import librosa

path = "/media/ml/data_ml/EEG/deepsleepnet/data_npy/SC4061E0.npz"

data = np.load(path)

x = data['x']
y = data['y']

fig_1 = plt.figure(figsize=(12, 6))
plt.plot(x[100, ...].ravel())
plt.title("EEG Epoch")
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.show()

fig_2 = plt.figure(figsize=(12, 6))
plt.plot(y.ravel())
plt.title("Sleep Stages")
plt.ylabel("Classes")
plt.xlabel("Time")
plt.show()