from pyspark.sql import SparkSession
from pymongo import MongoClient
import logger as lg

namenode_path = "namenode:9000"
spark = SparkSession.builder \
    .appName("Data Processing and MongoDB Integration") \
    .getOrCreate()

logger = lg.get_module_logger(__name__)
logger.debug("Get Spark data")
processed_data_path = f"hdfs://{namenode_path}/output/cleaned_data/*.csv"
df = spark.read.csv(processed_data_path, header=True)
pandas_df = df.toPandas()

logger.debug(pandas_df.head())

logger.debug("MongoDB Connection")
client = MongoClient("mongodb://mongodb:27017/")
db = client["customer_data_db"]
collection = db["processed_data"]
data = pandas_df.to_dict("records")
collection.insert_many(data)

logger.debug("Data successfully inserted into MongoDB.")

client.close()
spark.stop()
