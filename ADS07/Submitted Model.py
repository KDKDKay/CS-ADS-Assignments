import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self._feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self._feature_names]


# list of columns
unchanged_features = [
    "Medical_Keyword_15",
    "Medical_Keyword_23",
    "Medical_Keyword_48",
    "Medical_Keyword_3",
    "Employment_Info_3",
    "InsuredInfo_6",
    "Insurance_History_2",
    "Medical_History_4",
    "Medical_History_13",
    "Medical_History_16",
    "Medical_History_23",
    "Medical_History_30",
    "Medical_History_33",
    "Medical_History_39",
    "Medical_History_40",
]

constant_fill_features = ["Medical_History_32", "Medical_History_15"]

log_transform_features = [
    "Product_Info_4",
    "Ins_Age",
    "Wt",
    "BMI",
]

# add in unchanged features
unchanged_features_pipeline = Pipeline(
    [("unchanged_selector", FeatureSelector(unchanged_features))]
)


# create class and pipeline for discrete columns that have NaNs
class ConstantFill(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if y is None:
            return X.fillna(-9999)
        return X.fillna(-9999)


constant_fill_pipeline = Pipeline(
    [
        ("constant_selector", FeatureSelector(constant_fill_features)),
        ("constant_filler", ConstantFill()),
    ]
)


# create class and pipeline for numeric features to log transform
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for col in X.columns:
            X[col] = X[col].apply(lambda x: np.log(x + 1))

        if y is None:
            return X
        return X


log_transform_pipeline = Pipeline(
    [
        ("log_trans_selector", FeatureSelector(log_transform_features)),
        ("log_transformer", LogTransformer()),
    ]
)


def build_model():
    """This function builds a new model and returns it.
    The model should be implemented as a sklearn Pipeline object.
    Your pipeline needs to have two steps:
    - preprocessor: a Transformer object that can transform a dataset
    - model: a predictive model object that can be trained and generate predictions
    :return: a new instance of your model
    """

    preprocessor = FeatureUnion(
        [
            ("constant", constant_fill_pipeline),
            ("transform", log_transform_pipeline),
            ("unchanged", unchanged_features_pipeline),
        ]
    )

    model = lgb.LGBMClassifier(
        learning_rate=0.1,
        random_state=0,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=5,
        min_child_weight=9,
        n_estimators=100,
    )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])
