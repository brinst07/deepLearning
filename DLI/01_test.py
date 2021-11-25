import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import tensorflow as tf

m = -2  # -2 to start, change me please
b = 40  # 40 to start, change me please

# Sample data
x = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9])
y = np.array([10, 20, 25, 30, 40, 45, 40, 50, 60, 55])


tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
sgd = tf.keras.optimizers.SGD(lr=0.01)
tf.model.compile(loss='mse', optimizer=sgd)
tf.model.summary()
model = tf.model.fit(x, y, epochs=5000, verbose=1)
print(tf.model.predict(x))
m =tf.model.get_weights()[0][0]
b =tf.model.get_weights()[1][0]
y_hat = x * m + b
plt.plot(x,tf.model.predict(x), label='x겂')
plt.plot(x, y, '.')
plt.plot(x, y_hat, '-')
plt.show()

print("Loss:", np.sum((y - y_hat)**2)/len(x))