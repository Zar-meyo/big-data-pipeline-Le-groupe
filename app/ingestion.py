from datetime import date, datetime

from pyspark.sql import SparkSession
import pandas as pd
import logger as lg

spark = SparkSession.builder \
    .master("spark://spark:7077") \
    .appName("ingestion") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
    .getOrCreate()

logger = lg.get_module_logger(__name__)
logger.debug("pandas begin")
df = pd.read_csv("/usr/src/app/ecommerce_data_with_trends.csv")
logger.debug("pandas done")
logger.debug("spark begin")
df = spark.createDataFrame(df).repartition(4)
logger.debug("spark done")
df.show()