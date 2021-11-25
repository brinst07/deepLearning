from tensorflow.keras.datasets import mnist
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
x_train = x_train.reshape(60000, 784) / 255
x_valid = x_valid.reshape(10000, 784) / 255

num_categories = 10
y_train = keras.utils.to_categorical(y_train, num_categories)
y_valid = keras.utils.to_categorical(y_valid, num_categories)

model = Sequential()
# layer를 여러개 쌓으면 더 많은 매개변수를 제공할 수 있어서 정밀한 학습이 가능해진다.
# input Layer
model.add(Dense(units=512, activation='relu', input_shape=(784,)))
# hidden layer
model.add(Dense(units=512, activation='relu'))
# output layer
model.add(Dense(units=10, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, verbose=1, validation_data=(x_valid, y_valid))
