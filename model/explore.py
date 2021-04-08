import click
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.dates import date2num


@click.command()
@click.option("--data", type=click.Path(exists=True),
              default="data/data.csv")
def main(data):
    df = pd.read_csv(data)

    # Just look at the columns
    print(df.head())
    print(f"There are {len(df.columns)} columns in the dataset")

    # Check if article number coincides with article name
    # In fact it does, "art_no" = "art_name"
    print(df.groupby("art_no")["art_name"].agg(set))

    # Check if article number coincides numeric features
    # All features seem to be meaningful,
    # however "low_stock_warning" has zero variance for the following artiles
    # 2, 3, 11, 13, 17, 18, 19, 23, 25

    features = [
        "store_count",
        "unit_price_weekly",
        "unit_discount_weekly",
        "total_cust_count",
        "low_stock_warning",
    ]
    print(df.groupby("art_no")[features].agg(["mean", "std"]))

    # Check if article number coincides the target variable
    print(df.groupby("art_no")["sold_qty_units"].agg(["mean", "std"]))

    # Check if article number coincides with article name
    # In fact it does, "art_no" == "art_name"
    # So, the name and id are the same, and they differ across products
    for i in range(1, 4):
        print(f"Exploring category level {i}:")
        source, target = f"hierarchy_level{i}_id", f"hierarchy_level{i}_desc"
        print(df.groupby(source)[target].agg(set))
        print(df.groupby("art_no")[target].agg(set))

    # Check the target variable distribution
    # There are plenty of outliers in the dataset
    # Probably one should not rely on them during the training,
    # but let's keep them at the first iteration
    plt.figure(figsize=(20, 8))
    for name, group in df.groupby("art_name"):
        y, x = group["sold_qty_units"], group["date_of_day"]
        dates = date2num(x)
        plt.plot_date(dates, y, label=name)
    plt.ylim(0, 2000)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
