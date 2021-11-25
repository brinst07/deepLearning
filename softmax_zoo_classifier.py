import tensorflow as tf
import numpy as np

xy = np.loadtxt('./data/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7

# y_data(0~6)를 one_hot([1,0,0,....])으로 변경
y_one_hot = tf.keras.utils.to_categorical(y_data, nb_classes)
print("one_hot : ", y_one_hot)
# 순차함수 적용
tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=nb_classes, input_dim=16, activation="softmax"))
# 손실함수 : 모델이 최적에 사용하는 목적 함수 => categorical_crossentropy 사용
# opimizer : 훈련과정 설정, 최적화 알고리즘을 설정한다.
# metrics(평가지표) : 훈련을 모니터링 하기 위해 사용한다.
#                   분류에서는 accuracy, 회귀에서는 mse, rmse, r2 등이 있음
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=-0.1),
                 metrics=['accuracy'])
# 모델 정보 출력
tf.model.summary()

history = tf.model.fit(x_data, y_one_hot, epochs=1000)

# Single data test
test_data = np.array([[0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0]])  # expected prediction == 3 (feathers)
print(tf.model.predict(test_data), tf.model.predict_classes(test_data))

# Full x_data test
pred = tf.model.predict_classes(x_data)
for p, y in zip(pred, y_data.flatten()):
    print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
