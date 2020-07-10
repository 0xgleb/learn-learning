import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential(
    [ keras.layers.Dense(units=1, input_shape=[1]) ] # one input, one neuron
)

model.compile(
    optimizer='sgd', # stochastic gradient descent
    loss='mean_squared_error' # sum $ fmap (*2) differences
)

# example points for y = 3x + 1
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0,  1.0, 4.0, 7.0, 10, 13.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10]))
