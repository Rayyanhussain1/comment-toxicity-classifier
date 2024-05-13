import pandas as pd
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
import kerastuner as kt
from tensorflow.keras.layers import TextVectorization

df=pd.read_csv(r'C:\Users\Aun Electronic\Desktop\deep learning\train.csv')
print(df.head(5))
print(df.isnull().sum())
print(df.drop_duplicates())
df=df.drop(columns='id')
y=df.drop(columns='comment_text')
x=df['comment_text']
MAX_FEATURES=20000
vectorization=TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
vectorization.adapt(x.values)
vectorization_text=vectorization(x.values)

print(vectorization)
print(vectorization_text)

#MCSHBAP - map, chache, shuffle, batch, prefetch  from_tensor_slices, list_file
dataset = tf.data.Dataset.from_tensor_slices((vectorization_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8) # helps bottlenecks

train = dataset.take(int(len(dataset)*.7))
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))

model = Sequential()

model.add(Embedding(MAX_FEATURES+1, 32))

model.add(Bidirectional(LSTM(32, activation='tanh')))

model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))

model.add(Dense(6, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

history = model.fit(train, epochs=1, validation_data=val,batch_size=128)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()
