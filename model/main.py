import click
import pandas as pd
import numpy as np
from math import sqrt

from sklearn.metrics import mean_squared_error

from model.build import build_model


RANDOM_SEED = 137
np.random.seed(RANDOM_SEED)


def train_test_split_rounded(df, target=None, date_col="date_of_day", test_size=6):
    """
        Note: the test size is in month
    """
    pivot = df[date_col].max() - pd.tseries.offsets.MonthBegin(test_size)
    idx = df[date_col] < pivot

    # train, test
    if target:
        return df[idx], df[~idx], target[idx], target[~idx]
    else:
        return df[idx], df[~idx]


@click.command()
@click.option("--data", type=click.Path(exists=True),
              default="data/data.csv")
@click.option("--use-grid-cv", default=False)
def main(data, use_grid_cv):
    df = pd.read_csv(data)

    # convert to date format
    df.date_of_day = df.date_of_day.apply(pd.to_datetime)
    df['month'] = df.date_of_day.apply(lambda x: x.month)
    df['day'] = df.date_of_day.apply(lambda x: x.day)

    print("Build model")
    x = df.sort_values(by='date_of_day')
    X_tr, X_te, y_tr, y_te = train_test_split_rounded(x, x["sold_qty_units"],
                                                      test_size=6)

    print('train period: from {} to {}'.format(X_tr.date_of_day.min(), X_tr.date_of_day.max()))
    print('evaluation period: from {} to {}'.format(X_te.date_of_day.min(), X_te.date_of_day.max()))

    # Remove first examples with NaNs
    train_idx = ~np.isnan(X_tr.store_count_1_weeks_ago__10_halflife_ewm).values.reshape(-1)
    X_tr = X_tr[train_idx]
    y_tr = y_tr[train_idx]

    model = build_model()

    model.fit(X_tr, np.log(y_tr))  # force positive output
    print("Done fitting")

    print("Train set:")
    y_tr_predicted = np.exp(model.predict(X_tr))
    y_te_predicted = np.exp(model.predict(X_te))

    print("\tRMS:", sqrt(mean_squared_error(y_tr, y_tr_predicted)))

    print("Valid set:")
    print("\tRMS:", sqrt(mean_squared_error(y_te, y_te_predicted)))


if __name__ == '__main__':
    main()
