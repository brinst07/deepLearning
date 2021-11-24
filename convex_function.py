import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# 순차모델 사용
tf.model = tf.keras.Sequential()
# units : 출력 뉴런 수
# input_dim : 입력층의 뉴런 수
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))
# 경사하강법 옵티마지어 적용
sgd = tf.keras.optimizers.SGD(lr=0.1)
tf.model.compile(loss='mse', optimizer=sgd)

tf.model.summary()

history = tf.model.fit(x_train,y_train, epochs=200)

y_predict = tf.model.predict(np.array([100]))
print(y_predict)

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()