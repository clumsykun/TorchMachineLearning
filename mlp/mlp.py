"""
多层感知机
"""

import tensorflow as tf
import tensorflow.keras as keras
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class MLPRegressor(keras.Model):
    def __init__(self, n_property):
        super(MLPRegressor, self).__init__(name='MLP_Regressor')

        self.dense1 = keras.layers.Dense(32, activation='sigmoid', name='dense_1', input_shape=(n_property, ))
        self.dense2 = keras.layers.Dense(32, activation='sigmoid', name='dense_2')
        self.dense3 = keras.layers.Dense(32, activation='sigmoid', name='dense_3')
        self.dense4 = keras.layers.Dense(1, name='dense_4')

    def call(self, inputs):
        h1 = self.dense1(inputs)
        h2 = self.dense2(h1)
        h3 = self.dense3(h2)
        outputs = self.dense4(h3)
        return outputs


class MLPClassifier(keras.Model):
    def __init__(self, n_property):
        super(MLPClassifier, self).__init__(name='MLP_Classifier')
        self.dense1 = keras.layers.Dense(32, activation='relu', input_shape=(n_property,))
        self.dense2 = keras.layers.Dense(32, activation='relu')
        self.dense3 = keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        h1 = self.dense1(inputs)
        h2 = self.dense2(h1)
        outputs = self.dense3(h2)
        return outputs


def test_mlp_regressor():
    (x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()
    model = MLPRegressor(n_property=13)
    model.compile(
        optimizer=keras.optimizers.SGD(0.1),
        loss='mean_squared_error',  # keras.losses.mean_squared_error
        metrics=['mse'])

    model.fit(x_train, y_train, batch_size=50, epochs=100, validation_split=0.1, verbose=1)
    model.summary()
    model.evaluate(x_test, y_test)


def test_mlp_classifier():
    """
    accuracy = n_correct / n_total
    """
    whole_data = load_breast_cancer()
    x_data = whole_data.data
    y_data = whole_data.target

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=7)

    print(x_train.shape, ' ', y_train.shape)
    print(x_test.shape, ' ', y_test.shape)

    model = MLPClassifier(n_property=30)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.binary_crossentropy,
        metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=64, epochs=100, verbose=1)
    model.summary()
    model.evaluate(x_test, y_test)

if __name__ == "__main__":
    test_mlp_regressor()
    test_mlp_classifier()
