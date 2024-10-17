import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras import regularizers
from keras.optimizers import Adam
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression


def get_model(input_shape, use_scikit_learn=False, logreg=False):
    if logreg is False:
        if not use_scikit_learn:
            return DeepNeuralNetwork(input_shape, hidden_layer_sizes=(64, 32))
        else:
            return MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu")
    else:
        if not use_scikit_learn:
            return DeepNeuralNetwork(input_shape)
        else:
            return LogisticRegression()


class DeepNeuralNetwork():
    def __init__(self, input_shape, hidden_layer_sizes=(128, 32),
                 activation="relu", n_classes=2):
        reg = regularizers.L1L2(l1=1e-3, l2=1e-2)

        self.model = Sequential(
            [InputLayer(input_shape)] +
            [Dense(ls, activation=activation, kernel_regularizer=reg)
             for ls in hidden_layer_sizes] +
            [Dense(n_classes, kernel_regularizer=reg)])

        self.compile()

    def compile(self):
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = Adam()
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=[])

    def fit(self, X_train, y_train, n_epochs=100, early_stopping=False, verbose=False):
        if len(y_train) == 0:
            return

        callbacks = []
        if early_stopping is True:
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]

        self.model.fit(X_train, y_train, epochs=n_epochs, verbose=verbose, callbacks=callbacks)

    def __call__(self, X):
        return self.predict(X)

    def predict_proba(self, X):
        X_ = X
        if not isinstance(X, np.ndarray):
            X_ = np.array(X)

        return tf.nn.softmax(self.model(X_)).numpy()

    def predict(self, X):
        X_ = X
        if not isinstance(X, np.ndarray):
            X_ = np.array(X)

        y_pred = self.model(X_)
        return tf.argmax(tf.nn.softmax(y_pred), axis=1).numpy()
