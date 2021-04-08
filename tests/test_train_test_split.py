import pytest
import pandas as pd

from model.main import train_test_split_rounded


@pytest.fixture
def df():
    dates = pd.date_range('2018-01-01', '2020-01-01', freq='w')
    df = pd.DataFrame({"date_of_day": dates})
    return df


def test_splits_the_dataset(df, test_size=1):
    X_tr, X_te = train_test_split_rounded(df, target=None, test_size=1)
    assert X_tr["date_of_day"].max() < X_te["date_of_day"].min()
