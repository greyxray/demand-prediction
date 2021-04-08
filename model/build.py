from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, make_union
# from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb

'''
Building the model
'''


class PandasSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, records=False):
        self.columns = columns
        self.records = records

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.records:
            return X[self.columns].to_dict(orient="records")
        return X[self.columns]


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y):
        X = X.assign(target=y)
        self.frequencies = X.groupby(self.cols).size() / X.shape[0]
        return self

    def transform(self, X):
        transformed = X[self.cols].map(self.frequencies)
        return transformed.fillna(0.0).values.reshape(-1, 1)


def build_model():
    model = make_pipeline(
        make_union(
            make_pipeline(
                PandasSelector(["low_stock_warning"]),
            ),

            make_pipeline(
                PandasSelector(["hierarchy_level2_desc",
                                "hierarchy_level3_desc"]),
                OneHotEncoder(),
            ),

            make_pipeline(
                PandasSelector(["art_name"]),
                FrequencyEncoder("art_name"),
            ),

            make_pipeline(
                PandasSelector(["store_count", 'unit_price_weekly',
                                'unit_discount_weekly',
                                'unit_discount_weekly']),
                MinMaxScaler(),
            ),

            make_pipeline(
                ColumnTransformer(
                    [
                        ('minmax_weaks_ago', MinMaxScaler(), [
                            'sold_qty_units_1_weeks_ago',
                            # 'store_count_1_weeks_ago',
                            # 'total_cust_count_1_weeks_ago',
                            'sold_qty_units_1_weeks_ago__2_weeks_window_size',
                            # 'store_count_1_weeks_ago__2_weeks_window_size',
                            # 'total_cust_count_1_weeks_ago__2_weeks_window_size',
                            'sold_qty_units_1_weeks_ago__10_halflife_ewm',
                            # 'store_count_1_weeks_ago__10_halflife_ewm',
                            # 'total_cust_count_1_weeks_ago__10_halflife_ewm'
                            ]),
                    ],
                    remainder='drop'),
            ),

        ),


        # LinearRegression(),
        # SGDRegressor(random_state=0, max_iter=5)
        xgb.XGBRegressor(eta=0.32, n_estimators=11, seed=123)
    )
    return model
