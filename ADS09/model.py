from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class Preprocess(BaseEstimator, TransformerMixin):
    # create a class to makes words lowercase, remove punctuation, words
    # containing numbers and email addresses
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["text"] = X["text"].str.lower()
        X["text"] = X["text"].str.replace(
            r"[\w-]+@([\w-]+\.)+[\w-]+", ""
        )  # remove words with @/email addresses
        X["text"] = X["text"].str.replace(r"[^\w\s]", "")  # removes punctuation
        X["text"] = X["text"].str.replace(r"\w*\d\w*", "")  # removes words with numbers
        # removes carriage returns line breaks, tabs, replace with space
        X["text"] = X["text"].str.replace(r"/(\r\n)+|\r+|\n+|\t+/i", " ")

        if y is None:
            return X
        return X, y


vectoriser = ColumnTransformer([("tfidvectorise", TfidfVectorizer(), "text")])


def build_model():
    """This function builds a new model and returns it.

    The model should be implemented as a sklearn Pipeline object.

    Your pipeline needs to have two steps:
    - preprocessor: a Transformer object that can transform a dataset
    - model: a predictive model object that can be trained and generate predictions

    :return: a new instance of your model
    """

    preprocessor = Pipeline([("process", Preprocess()), ("vectorise", vectoriser)])

    preprocessor = ColumnTransformer(
        [
            (
                "tfidvectorise",
                TfidfVectorizer(stop_words="english", ngram_range=(1, 2)),
                "text",
            )
        ]
    )
    return Pipeline(
        [("preprocessor", preprocessor), ("model", MultinomialNB(alpha=0.04))]
    )
