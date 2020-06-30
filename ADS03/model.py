import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


def preprocess(df):
    """This function takes a dataframe and preprocesses it so it is
    ready for the training stage.

    The DataFrame contains columns used for training (features)
    as well as the target column.

    It also contains some rows for which the target column is unknown.
    Those are the observations you will need to predict for KATE
    to evaluate the performance of your model.

    Here you will need to return the training set: X and y together
    with the preprocessed evaluation set: X_eval.

    Make sure you return X_eval separately! It needs to contain
    all the rows for evaluation -- they are marked with the column
    evaluation_set. You can easily select them with pandas:

         - df.loc[df.evaluation_set]

    For y you can either return a pd.DataFrame with one column or pd.Series.

    :param df: the dataset
    :type df: pd.DataFrame
    :return: X, y, X_eval
    """

    # create regions
    def region(x):
        if x in ("US", "CA", "MX"):
            return "USA"
        elif x in ("CA", "MX"):
            return "NA"
        elif x in ("NZ", "SG", "HK"):
            return "Other"
        else:
            return "EU"

    df["region"] = df.country.apply(region)

    # create USD goal
    df["goal_usd"] = df["goal"] * df["static_usd_rate"]

    # normalise USD goal
    def normalize(column):
        upper = column.max()
        lower = column.min()
        y = (column - lower) / (upper - lower)
        return y

    # transform goal_usd
    df["goal_norm"] = np.log(df.goal_usd + 1)
    df["goal_trans"] = normalize(df.goal_norm)

    # add time functions
    df["time_open"] = df.deadline - df.launched_at
    df["time_to_launch"] = df.launched_at - df.created_at

    # create test_df
    test_df = df[
        [
            "goal_trans",
            "region",
            "time_open",
            "time_to_launch",
            "category",
            "state",
            "evaluation_set",
        ]
    ]

    # label encode the region
    label = LabelEncoder()
    label.fit(test_df.region.unique())
    test_df.loc[:, "region_enc"] = pd.Series(label.transform(test_df.region))

    # get_dummies for the category
    df["category"] = pd.Categorical(test_df["category"])
    dfdummies = pd.get_dummies(df["category"], prefix="cat")
    test_df = pd.concat([test_df, dfdummies], axis=1)

    # create test sets
    msk_eval = test_df.evaluation_set
    X = test_df[~msk_eval].drop(
        ["category", "state", "region", "evaluation_set"], axis=1
    )
    y = test_df[~msk_eval]["state"]
    X_eval = test_df[msk_eval].drop(
        ["category", "state", "region", "evaluation_set"], axis=1
    )

    return X, y, X_eval


def train(X, y):
    """Trains a new model on X and y and returns it.

    :param X: your processed training data
    :type X: pd.DataFrame
    :param y: your processed label y
    :type y: pd.DataFrame with one column or pd.Series
    :return: a trained model
    """
    model = RandomForestClassifier()
    model.fit(X, y)
    return model


def predict(model, X_test):
    """This functions takes your trained model as well
    as a processed test dataset and returns predictions.

    On KATE, the processed test dataset will be the X_eval you built
    in the "preprocess" function. If you're testing your functions locally,
    you can try to generate predictions using a sample test set of your
    choice.

    This should return your predictions either as a pd.DataFrame with one column
    or a pd.Series

    :param model: your trained model
    :param X_test: a processed test set (on KATE it will be X_eval)
    :return: y_pred, your predictions
    """
    y_pred = model.predict(X_test)
    return y_pred
