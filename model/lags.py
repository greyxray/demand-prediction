
TARGET_COLUMNS = [
    "sold_qty_units",
    "store_count",
    "total_cust_count",
]

def lag_columns(
    df,
    cols=TARGET_COLUMNS,
    date_col="date_of_day",
    group_col="art_no",  # TODO: make features per category etc.
    n_intervals=1,  # TODO: make features a month ago, year ago etc.
):
    sdf = df.sort_values([date_col, group_col])
    groups = sdf.groupby(group_col)
    # breakpoint()
    lag_col_names = []
    for col in cols:
        lag_col = f"{col}_{n_intervals}_weeks_ago"
        sdf[lag_col] = groups[col].shift(n_intervals)
        lag_col_names.append(lag_col)

    return sdf, lag_col_names


# The functions below should be called only on the outputs of lag_columns func.


def lag_average(
    df,
    n=2,  # window, size -- the larger window, the higher is regularization
    cols=None,
    date_col="date_of_day",
    group_col="art_no",  # TODO: make features per category etc.
):
    sdf = df.sort_values([date_col, group_col])
    groups = sdf.groupby(group_col)
    for col in cols:
        cname = f"{col}__{n}_weeks_window_size"
        sdf[cname] = groups[col].transform(lambda x: x.rolling(n, 1).mean())

    return sdf


# Ads more impact to the last observation

def lag_ewm(
    df,
    halflife=10,  # The lower, the higher is the impact of the last entry
    cols=None,
    date_col="date_of_day",
    group_col="art_no",  # TODO: make features per category etc.
    n_intervals=1,  # TODO: make features a month ago, year ago etc.
):
    sdf = df.sort_values([date_col, group_col])
    groups = sdf.groupby(group_col)
    for col in cols:
        cname = f"{col}__{halflife}_halflife_ewm"
        sdf[cname] = groups[col].transform(
            lambda x: x.ewm(halflife=halflife).mean())
    return sdf
