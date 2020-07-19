from keras.layers.core import Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dropout
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # reshape to a vector and normalise
        n_rows, n_cols = X.shape[1:]
        n_dims = 1  # as black and white images
        X = X.reshape(X.shape[0], n_rows, n_cols, n_dims)
        X /= 255

        if y is None:
            return X

        # categorise the y values for keras
        y = np_utils.to_categorical(y, 4)
        return X, y


def keras_builder():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1)))
    model.add(
        Conv2D(
            64,
            (5, 5),
            activation="relu",
            padding="same",
            input_shape=(28, 28, 1),
            name="conv2_1",
        )
    )
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (5, 5), activation="relu", name="conv2_2"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(4, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def build_model():
    """This function builds a new model and returns it.

    The model should be implemented as a sklearn Pipeline object.

    Your pipeline needs to have two steps:
    - preprocessor: a Transformer object that can transform a dataset
    - model: a predictive model object that can be trained and generate predictions

    :return: a new instance of your model
    """
    preprocessor = Preprocessor()

    model = KerasClassifier(build_fn=keras_builder, batch_size=32, epochs=15)

    return Pipeline([("preprocessor", preprocessor), ("model", model)])
