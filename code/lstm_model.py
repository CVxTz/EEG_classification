from models import get_model_lstm
import numpy as np
from utils import gen, chunker, WINDOW_SIZE, rescale_array
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, classification_report
from glob import glob
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm


base_path = "/media/ml/data_ml/EEG/deepsleepnet/data_npy"

files = sorted(glob(os.path.join(base_path, "*.npz")))
train_val, test = files[:-6], files[-6:]

train, val = train_test_split(train_val, test_size=0.1, random_state=1337)

train_dict = {k: np.load(k) for k in train}
test_dict = {k: np.load(k) for k in test}
val_dict = {k: np.load(k) for k in val}

model = get_model_lstm()

file_path = "lstm_model.h5"
# model.load_weights(file_path)

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=4, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=2, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

model.fit_generator(gen(train_dict, aug=False), validation_data=gen(val_dict), epochs=100, verbose=2,
                    steps_per_epoch=1000, validation_steps=300, callbacks=callbacks_list)
model.load_weights(file_path)


preds = []
gt = []

for record in tqdm(test_dict):
    all_rows = test_dict[record]['x']
    for batch_hyp in chunker(range(all_rows.shape[0])):


        X = all_rows[min(batch_hyp):max(batch_hyp), ...]
        Y = test_dict[record]['y'][min(batch_hyp):max(batch_hyp)]

        X = np.expand_dims(X, 0)

        X = rescale_array(X)

        Y_pred = model.predict(X)
        Y_pred = Y_pred.argmax(axis=-1).ravel().tolist()

        gt += Y.ravel().tolist()
        preds += Y_pred



f1 = f1_score(gt, preds, average="macro")

print("Seq Test f1 score : %s "% f1)

acc = accuracy_score(gt, preds)

print("Seq Test accuracy score : %s "% acc)

print(classification_report(gt, preds))