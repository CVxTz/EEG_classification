from models import get_model_cnn_crf
import numpy as np
from utils import gen, chunker, WINDOW_SIZE, rescale_array
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, classification_report
from glob import glob
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt


base_path = "/media/ml/data_ml/EEG/deepsleepnet/data_npy"

files = sorted(glob(os.path.join(base_path, "*.npz")))

ids = list(set([x.split("/")[-1][:5] for x in files]))
list_f1 = []
list_acc = []
preds = []
gt = []
for id in ids:
    test_ids = {id}
    train_ids = set([x.split("/")[-1][:5] for x in files]) - test_ids

    train_val, test = [x for x in files if x.split("/")[-1][:5] in train_ids],\
                      [x for x in files if x.split("/")[-1][:5] in test_ids]

    train, val = train_test_split(train_val, test_size=0.1, random_state=1337)

    train_dict = {k: np.load(k) for k in train}
    test_dict = {k: np.load(k) for k in test}
    val_dict = {k: np.load(k) for k in val}

    model = get_model_cnn_crf(lr=0.0001)

    file_path = "cnn_crf_model_20_folds.h5"
    # model.load_weights(file_path)

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=20, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=5, verbose=2)
    callbacks_list = [checkpoint, redonplat]  # early

    model.fit_generator(gen(train_dict, aug=False), validation_data=gen(val_dict), epochs=40, verbose=2,
                        steps_per_epoch=1000, validation_steps=300, callbacks=callbacks_list)
    model.load_weights(file_path)




    for record in tqdm(test_dict):
        all_rows = test_dict[record]['x']
        record_y_gt = []
        record_y_pred = []
        for batch_hyp in chunker(range(all_rows.shape[0])):


            X = all_rows[min(batch_hyp):max(batch_hyp)+1, ...]
            Y = test_dict[record]['y'][min(batch_hyp):max(batch_hyp)+1]

            X = np.expand_dims(X, 0)

            X = rescale_array(X)

            Y_pred = model.predict(X)
            Y_pred = Y_pred.argmax(axis=-1).ravel().tolist()

            gt += Y.ravel().tolist()
            preds += Y_pred

            record_y_gt += Y.ravel().tolist()
            record_y_pred += Y_pred


f1 = f1_score(gt, preds, average="macro")

acc = accuracy_score(gt, preds)

print("acc %s, f1 %s"%(acc, f1))

