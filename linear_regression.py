import numpy as np
import tensorflow as tf

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# keras? => Tensorflow 위에서 동작하는 프레임워크
# Tensorflow보다 훨씬 간단하게 사용이 가능하다.
# Suquential 모델은 순차적으로 레이어 층을 더해주기 떄문에 순차모델이라고 불린다.
tf.model = tf.keras.Sequential()
# Dense란 신경망을 만드는 것이다.
# input을 넣었을 때, output으로 바꿔주는 중간 다리이다.
# hidden layer를 생성함, 사람으로 따지면 뇌다.
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))
# optimizer는 keras 모델을 컴파일하기 위해 필요한 두 개의 매개변수 중 하나이다.
# 확률적 경사 하강법(SGD) 옵티마이저
# lr: 0보다 크거나 같은 float 값. 학습률.
# momentum: 0보다 크거나 같은 float 값. SGD를 적절한 방향으로 가속화하며, 흔들림(진동)을 줄여주는 매개변수입니다.
# decay: 0보다 크거나 같은 float 값. 업데이트마다 적용되는 학습률의 감소율입니다.
# nesterov: 불리언. 네스테로프 모멘텀의 적용 여부를 설정합니다.
sgd = tf.keras.optimizers.SGD(lr=0.1)
# 객체를 만들기 위한 두가지 매개변수를 넣어서 객체를 생성한다.
# loss : 손실함수
# optimizer
tf.model.compile(loss='mse', optimizer=sgd)
# 모델의 구조를 요약해 출력해줍니다.
tf.model.summary()
# fit => keras 학습 함수
# fit(X,Y,batch_size=100, epochs=10)
# X : 입력 데이터, Y : 결과(Label 값) 데이터, batch_size : 한번에 학습할 때 사용하는 데이터 개수, epochs : 학습 데이터 반복 횟수
tf.model.fit(x_train, y_train, epochs=200)

# predict는 데이터 배치의 각 이미지에 대해 하나의 목록씩 목록의 목록을 반환합니다.
# y_predict = tf.model.predict(np.array([5,4]))
y_predict = tf.model.predict(np.arange(0, 10))
print(y_predict)