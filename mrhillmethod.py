import os 
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import tensorflow as tf

data = pd.read_csv("all_classifcation_and_seqs_aln.csv")
data = data.dropna()
X = data["sequence"]
y = data["species"]

encoder = LabelEncoder()


def encodeSequences(seqs):
    res = []
    for i in seqs:
        seq = list(i)
        res.append(np.array(encodeseq(seq)))
    return res

def encodeseq(seq):
    missionary = {
        "-":0,
        "A":1,
        "T":2,
        "C":2,
        "G":3
    }
    res = []
    for i in seq:
        res.append(missionary[i])
    return res


X = np.array(encodeSequences(X))
y = np.array(encoder.fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(np.unique(X_train[0]))

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[(X_train.shape)[1]]),
    tf.keras.layers.Dense(11, activation='relu'),
    tf.keras.layers.Dense(46),
    tf.keras.layers.Softmax()
])

# TODO set your learning rate
lr = 0.00002

#TODO Compile your model with a selected optimizer and loss function

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    metrics=["accuracy"]
)

# TODO: fit your model with X_train and Y_train
history = model.fit(X_train, y_train, epochs=500, validation_data=(X_test,y_test))

df = pd.DataFrame(history.history)['val_loss']
px.scatter(df).show()
print("i been running into you in my head")