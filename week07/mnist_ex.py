from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy
# data load
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("train_images.max :", numpy.max(train_images))
print("="*30)
print("shape")
print("="*30)
print("train_images:", train_images.shape)
print("train_labels:", train_labels.shape)
print("test_images:", test_images.shape)
print("test_labels:", test_labels.shape)
print("="*30)
print("type")
print("="*30)
print("train_images:", type(train_images))
print("train_labels:", type(train_labels))
print("test_images:", type(test_images))
print("test_labels:", type(test_labels))

# data reshape and normalization
L, W, H = train_images.shape
train_images = train_images.reshape((60000, W * H)) # (-1,W*H)
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, W * H))  # (-1,W*H)
test_images = test_images.astype('float32') / 255

# binary target (one hot encoding)
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("="*30)
print("after to_categorical")
print("="*30)
print("tl:", train_labels[0])
print("ttl:", test_labels[0])
print("train_labels:", train_labels.shape)
print("test_labels:", test_labels[0].shape)
print("train_labels type:", type(train_labels))
print("test_labels type:", type(test_labels))

# network design
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(W * H,)))
network.add(layers.Dense(10, activation='softmax'))

# setting optimizer and loss function
network.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])

# fitting
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# evaluate on test set
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
