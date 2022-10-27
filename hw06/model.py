from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

iris = datasets.load_iris()
X=iris.data
Y=iris.target
scaler=StandardScaler()
X_=scaler.fit_transform(X)
enc=OneHotEncoder()
Y_ = enc.fit_transform(Y.reshape(-1,1)).toarray()
X_train, X_test, Y_train, Y_test = train_test_split(X_, Y_, test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


from tensorflow.keras import layers, models, optimizers

model = models.Sequential()
model.add(layers.Dense(5, activation='relu', input_shape=(4,)))
model.add(layers.Dense(5, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs=50)
score = model.evaluate(X_test, Y_test)
print('accuracy:', score[1])

from matplotlib import pyplot as ppt

print("Problem 3")
ppt.plot(history.history['accuracy'])
ppt.show()

print("Problem 4")
print(Y_test[0:5,])
print(np.argmax(Y_test[0:5,], axis=-1))

print("Problem 5")
print(model.predict(X_test[0:5,]))
## tensorflow 2.6 이후, predict_classes가 사라져서 아래의 코드로 대체
## https://leunco.tistory.com/16
print(model.predict(X_test[0:5,]).argmax(axis=-1))