import logger as lg
from pyspark.sql.functions import split, sum, avg, col, month, dayofweek
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = lg.get_module_logger(__name__)


def get_separate_categories(spark_df):
    logger.debug("Separating main and sub product category")
    final_df = spark_df.withColumn('main_category', split(spark_df['category'], ' > ').getItem(0)) \
        .withColumn('sub_category', split(spark_df['category'], ' > ').getItem(1))
    return final_df


def get_day_and_month(spark_df):
    logger.debug("Getting month and day of week")
    return spark_df.withColumn("month", month("timestamp")).withColumn("weekday", dayofweek("timestamp"))


def get_top_spenders(spark_df, dir_path):
    logger.debug("Get top customers by total spent")
    top_spenders = spark_df.groupBy('customer_id', 'customer_name', 'customer_type') \
        .agg(sum('total_amount').alias('total_spent')) \
        .orderBy(col('total_spent').desc()) \
        .limit(20)
    top_spenders.toPandas().to_csv(f"{dir_path}/top_spenders.csv")


def client_mean_purchase_per_category(spark_df, dir_path):
    logger.debug("Plotting mean purchases per category")
    # Get mean purchases per main category
    purchase_sum_df = spark_df.groupBy('customer_type', 'customer_id', 'main_category').agg(
        sum('quantity').alias('quantity_sum'))
    mean_main_df = purchase_sum_df.groupBy('customer_type', 'main_category').agg(
        avg('quantity_sum').alias('mean_quantity'))

    df_pandas = mean_main_df.toPandas()
    plt.figure()
    sns.barplot(x="main_category", y="mean_quantity", hue="customer_type", data=df_pandas)
    plt.xlabel('Product main category')
    plt.ylabel('Mean amount')
    plt.title("Mean product purchases per customer type")
    plt.xticks(rotation=45)
    plt.savefig(f"{dir_path}/mean_main_category_purchases.png", bbox_inches='tight')
    plt.close()


def client_mean_purchase_sub_category(spark_df, dir_path):
    logger.debug("Plotting mean purchases per sub category")
    grouped_df = spark_df.groupBy('customer_type', 'customer_id', 'main_category', 'sub_category') \
        .agg(sum('quantity').alias('quantity_sum'))
    mean_category_df = grouped_df.groupBy('customer_type', 'main_category', 'sub_category').agg(
        avg('quantity_sum').alias('mean_quantity'))

    categories = spark_df.select('main_category').distinct().toPandas()['main_category'].to_list()
    for main_category in categories:
        category_value = main_category.replace(' ', '_')
        logger.debug(f"Plotting for category {category_value}")
        category_df = mean_category_df.where(mean_category_df["main_category"] == main_category)

        pandas_df = category_df.toPandas()
        plt.figure()
        sns.barplot(x="sub_category", y="mean_quantity", hue="customer_type", data=pandas_df)
        plt.xlabel(f"{main_category} sub category")
        plt.ylabel("Mean amount")
        plt.title("Mean product purchases per customer type")
        plt.xticks(rotation=45)
        plt.savefig(f"{dir_path}/mean_{category_value}_purchases.png", bbox_inches='tight')
        plt.close()


def get_month_trends(spark_df, dir_path):
    logger.debug("Plotting main category purchases per month")
    avg_per_category_df = spark_df.groupBy('customer_id', 'main_category', 'month') \
        .agg(sum('quantity').alias('quantity_sum')) \
        .groupBy('main_category', 'month') \
        .agg(avg('quantity_sum').alias('mean_quantity'))

    df_pandas = avg_per_category_df.toPandas()
    plt.figure()
    sns.barplot(x="month", y="mean_quantity", hue="main_category", data=df_pandas)
    plt.xlabel('Month number')
    plt.ylabel('Mean amount')
    plt.title("Mean product purchases per month")
    plt.savefig(f"{dir_path}/mean_purchases_per_month.png", bbox_inches='tight')
    plt.close()

    avg_per_sub_df = spark_df.groupBy('customer_id', 'main_category', 'sub_category', 'month') \
        .agg(sum('quantity').alias('quantity_sum')) \
        .groupBy('main_category', 'sub_category', 'month') \
        .agg(avg('quantity_sum').alias('mean_quantity'))

    categories = spark_df.select('main_category').distinct().toPandas()['main_category'].to_list()
    for main_category in categories:
        category_value = main_category.replace(' ', '_')
        logger.debug(f"Plotting for category {main_category}")
        category_df = avg_per_sub_df.where(avg_per_sub_df["main_category"] == main_category)

        pandas_df = category_df.toPandas()
        plt.figure()
        sns.barplot(x="month", y="mean_quantity", hue="sub_category", data=pandas_df)
        plt.xlabel("Month number")
        plt.ylabel("Mean amount")
        plt.title("Mean product purchases per month")
        plt.savefig(f"{dir_path}/mean_{category_value}_purchases_per_month.png", bbox_inches='tight')
        plt.close()


def get_weekday_trends(spark_df, dir_path):
    logger.debug("Plotting main category purchases per month")
    avg_per_category_df = spark_df.groupBy('customer_id', 'main_category', 'weekday') \
        .agg(sum('quantity').alias('quantity_sum')) \
        .groupBy('main_category', 'weekday') \
        .agg(avg('quantity_sum').alias('mean_quantity'))

    df_pandas = avg_per_category_df.toPandas()
    plt.figure()
    sns.barplot(x="weekday", y="mean_quantity", hue="main_category", data=df_pandas)
    plt.xlabel('Weekday number')
    plt.ylabel('Mean amount')
    plt.title("Mean product purchases per weekday")
    plt.savefig(f"{dir_path}/mean_purchases_per_weekday.png", bbox_inches='tight')
    plt.close()

    avg_per_sub_df = spark_df.groupBy('customer_id', 'main_category', 'sub_category', 'weekday') \
        .agg(sum('quantity').alias('quantity_sum')) \
        .groupBy('main_category', 'sub_category', 'weekday') \
        .agg(avg('quantity_sum').alias('mean_quantity'))

    categories = spark_df.select('main_category').distinct().toPandas()['main_category'].to_list()
    for main_category in categories:
        category_value = main_category.replace(' ', '_')
        logger.debug(f"Plotting for category {main_category}")
        category_df = avg_per_sub_df.where(avg_per_sub_df["main_category"] == main_category)

        pandas_df = category_df.toPandas()
        plt.figure()
        sns.barplot(x="weekday", y="mean_quantity", hue="sub_category", data=pandas_df)
        plt.xlabel("Month number")
        plt.ylabel("Mean amount")
        plt.title("Mean product purchases per weekday")
        plt.savefig(f"{dir_path}/mean_{category_value}_purchases_per_weekday.png", bbox_inches='tight')
        plt.close()


def run_spark_analysis(spark_df, is_processed=True):
    dir_path = "../analysis/spark_analysis"
    if not is_processed:
        spark_df = get_separate_categories(spark_df)
        spark_df = get_day_and_month(spark_df)
    logger.debug("Creating spark analysis directory")
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    get_top_spenders(spark_df, dir_path)
    client_mean_purchase_per_category(spark_df, dir_path)
    client_mean_purchase_sub_category(spark_df, dir_path)
    get_month_trends(spark_df, dir_path)
    get_weekday_trends(spark_df, dir_path)
    logger.debug("Spark analysis over")
