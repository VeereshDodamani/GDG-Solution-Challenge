import pandas as pd
import numpy as np
import seaborn as sns
from tensorflow import keras

df_train = pd.read_csv('DiseaseTraining.csv')
df_test = pd.read_csv('DiseaseTesting.csv')

df_train.head()
df_train.isnull().sum()
df_train['prognosis'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_train['prognosis'] = le.fit_transform(df_train['prognosis'])
df_train.head()
df_train['prognosis'].max()

df_test['prognosis'] = le.transform(df_test['prognosis'])
df_test.head(20)
x_train = df_train.drop(columns='prognosis')
y_train = df_train['prognosis']

x_test = df_test.drop(columns='prognosis')
y_test = df_test['prognosis']

from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(units=32, activation='relu', input_shape=(x_train.shape[-1],)))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=41, activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=1000, epochs=50, verbose=2)
prediction = model.predict(x_test)
print(prediction[:1])

prediction = np.argmax(prediction, axis=-1)
print(prediction[:5])

print(y_test[:5])
from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y_test, prediction))
