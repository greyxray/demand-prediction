import random
import pytest
import pandas as pd


@pytest.fixture()
def data():
    entries = []
    for art_no in range(1, 26):
        for date_of_day in pd.date_range('2018-01-01', '2020-01-01', freq='w'):
            entry = {
                'art_no': art_no,
                'art_name': f"article {art_no}",
                'hierarchy_level1_id': 1,
                'hierarchy_level1_desc': f'category {0}',
                'hierarchy_level2_id': art_no % 4,
                'hierarchy_level2_desc': f'sub category {art_no % 4}',
                'hierarchy_level3_id': art_no % 5,
                'hierarchy_level3_desc': f'sub sub category {art_no % 5}',
                'date_of_day': date_of_day,
                # Note it's possible to solve this problem probabilistically
                # we have just to determine the right priors,
                # the distributions here make no sense :)
                'store_count': random.randint(0, 200),
                'unit_price_weekly': random.uniform(0.5, 20),
                'unit_discount_weekly': random.uniform(0, 1),
                'total_cust_count': random.uniform(10, 500),
                'low_stock_warning': random.betavariate(0.5, 0.5),
                'sold_qty_units': random.uniform(10, 500),
            }
            entries.append(entry)
    df = pd.DataFrame(entries)
    # emulate preprocessing
    df["date"] = df["date_of_day"]
    return df
