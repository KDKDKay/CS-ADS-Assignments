from fbprophet import Prophet


def preprocess(df):
    """This function takes a dataframe and preprocesses it so it is
    ready for the training stage.

    The DataFrame contains the time axis and the target column.

    It also contains some rows for which the target column is unknown.
    Those are the observations you will need to predict for KATE
    to evaluate the performance of your model.

    Here you will need to return the training time serie: ts together
    with the preprocessed evaluation time serie: ts_eval.

    Make sure you return ts_eval separately! It needs to contain
    all the rows for evaluation -- they are marked with the column
    evaluation_set. You can easily select them with pandas:

         - df.loc[df.evaluation_set]


    :param df: the dataset
    :type df: pd.DataFrame
    :return: ts, ts_eval
    """

    # transform df for Prophet readability
    df['y'] = df.consumption
    df['ds'] = df.day
    df = df.drop(['day', 'consumption'], axis=1)

    # Save msk to split data later
    msk_eval = df.evaluation_set
    df.drop("evaluation_set", axis=1, inplace=True)

    # Split training/test data
    ts = df[~msk_eval]
    ts_eval = df[msk_eval]

    return ts, ts_eval


def train(ts):
    """Trains a new model on ts and returns it.

    :param ts: your processed training time serie
    :type ts: pd.DataFrame
    :return: a trained model
    """

    # Train Prophet model
    model = Prophet(growth='linear', daily_seasonality=0)
    model.fit(ts)
    return model


def predict(model, ts_test):
    """This functions takes your trained model as well
    as a processed test time serie and returns predictions.

    On KATE, the processed testt time serie will be the ts_eval you built
    in the "preprocess" function. If you're testing your functions locally,
    you can try to generate predictions using a sample test set of your
    choice.

    This should return your predictions either as a pd.DataFrame with one column
    or a pd.Series

    :param model: your trained model
    :param ts_test: a processed test time serie (on KATE it will be ts_eval)
    :return: y_pred, your predictions
    """

    # set prediction parameters and find total preds
    df_dates = model.make_future_dataframe(periods=27, include_history=True)
    model_predictions = model.predict(df_dates)

    # extrapolate preds
    preds = model_predictions.tail(27)['yhat']
    return preds
