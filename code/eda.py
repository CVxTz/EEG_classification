import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import collections
import librosa

filename = "../input/challenge_data.h5"

f = h5py.File(filename, "r")




def preprocess_audio_mel_T(audio, sample_rate=250, n_mels=59):

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels= n_mels, hop_length=128)
    mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40)/40

    return mel_db.T

x = f['2']['channel1'][2000:(2000+250*30)]

out = preprocess_audio_mel_T(x)
print(out.shape)
plt.imshow(out)
plt.colorbar()
plt.show()
