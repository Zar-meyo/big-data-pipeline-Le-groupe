import logger as lg
from pyspark.sql.functions import split, month, dayofweek

logger = lg.get_module_logger(__name__)


def process_data(spark_df):
    logger.debug("Processing data")
    logger.debug("Removing missing values")
    df = spark_df.dropna()
    logger.debug("Removing duplicate values")
    df = spark_df.distinct()

    logger.debug("Splitting main and sub categories of products")
    df = df.withColumn('main_category', split(spark_df['category'], ' > ').getItem(0)) \
        .withColumn('sub_category', split(spark_df['category'], ' > ').getItem(1))
    logger.debug("Get month and day of week from timestamp")
    df = df.withColumn("month", month("timestamp")).withColumn("weekday", dayofweek("timestamp"))
    return df
