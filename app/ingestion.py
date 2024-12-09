import pandas as pd
import logger as lg


def get_spark_df(spark):
    logger = lg.get_module_logger(__name__)
    logger.debug("pandas begin")
    df = pd.read_csv("/usr/src/app/ecommerce_data_with_trends.csv")
    logger.debug("pandas done")
    logger.debug("spark begin")
    df = spark.createDataFrame(df).repartition(4)
    logger.debug("spark done")
    return df
