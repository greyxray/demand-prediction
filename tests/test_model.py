from model.lags import lag_columns, lag_average, lag_ewm
from model.build import build_model


def test_baseline(data):
    # Just check if the pipeline works
    x, lags = lag_columns(data)
    x = lag_average(x, cols=lags)
    x = lag_ewm(x, cols=lags)

    model = build_model().fit(x, x["sold_qty_units_1_weeks_ago"])
    model.predict(x)
