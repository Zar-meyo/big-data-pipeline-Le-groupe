import logger as lg
from pyspark.sql.functions import split, sum, avg, col
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = lg.get_module_logger(__name__)

def get_separate_categories(spark_df):
    logger.debug("Separating main and sub product category")
    final_df = spark_df.withColumn('main_category', split(spark_df['category'], ' > ').getItem(0))\
            .withColumn('sub_category', split(spark_df['category'], ' > ').getItem(1))
    return final_df

def get_top_spenders(spark_df, dir_path):
    logger.debug("Get top customers by total spent")
    top_spenders = spark_df.groupBy('customer_id', 'customer_name', 'customer_type')\
            .agg(sum('total_amount').alias('total_spent'))\
            .orderBy(col('total_spent').desc())\
            .limit(20)
    top_spenders.write.mode("overwrite").csv(f"{dir_path}/top_spenders.csv", header=True)

def client_mean_purchase_per_category(spark_df, dir_path):
    logger.debug("Plotting mean purchases per category")
    # Get mean purchases per main category
    purchase_sum_df = spark_df.groupBy('customer_type', 'customer_id', 'main_category').agg(sum('quantity').alias('quantity_sum'))
    mean_main_df = purchase_sum_df.groupBy('customer_type', 'main_category').agg(avg('quantity_sum').alias('mean_quantity'))

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
    grouped_df = spark_df.groupBy('customer_type', 'customer_id', 'main_category', 'sub_category')\
            .agg(sum('quantity').alias('quantity_sum'))
    mean_category_df = grouped_df.groupBy('customer_type', 'main_category', 'sub_category').agg(avg('quantity_sum').alias('mean_quantity'))

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

def run_spark_analysis(spark_df, is_processed=True):
    dir_path = "../analysis/spark_analysis"
    if not is_processed:
        spark_df = get_separate_categories(spark_df)
    logger.debug("Creating spark analysis directory")
    Path(dir_path).mkdir(parents=True, exist_ok=True)

    #get_top_spenders(spark_df, dir_path)
    client_mean_purchase_per_category(spark_df, dir_path)
    client_mean_purchase_sub_category(spark_df, dir_path)
    logger.debug("Spark analysis over")
