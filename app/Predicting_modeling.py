from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, count, sum, avg, count, collect_set, expr, hour, dayofweek, month, to_date, year, \
    lead, when, datediff, max, first, unix_timestamp, lag, collect_list, size
from pyspark.ml.feature import StringIndexer, VectorAssembler
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

import logger as lg


def ml_prediction(spark_df, spark):
    dir_path = "../analysis/ml_prediction"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger = lg.get_module_logger(__name__)
    spark_df.show(5)
    # df.printSchema()

    ###################################################################
    # First, we will do random forest for future purchase prediction
    # Try to predict if a customer will purchase on a hour

    # I tested on a month and day, but the result prediction and accuracy was 1
    # because it seems like all people from the dataset do a purchase
    # in the next month or next day

    # need to upgrade this model
    ###################################################################
    logger.debug("Begin dataset for Random Forest Classifier")
    # Do column of multiple time data to use
    spark_df = spark_df.withColumn("date", to_date(col("timestamp")))
    spark_df = spark_df.withColumn("year", year(col("date")))
    spark_df = spark_df.withColumn("month", month(col("date")))
    spark_df = spark_df.withColumn("day_of_week", dayofweek(col("date")))
    spark_df = spark_df.withColumn("hour_of_day", hour(col("timestamp")))

    window = Window.partitionBy("customer_id").orderBy("timestamp")
    spark_df = spark_df.withColumn("date_next_purchase", lead("timestamp", 1).over(window))
    spark_df = spark_df.withColumn("hours_between", (
            unix_timestamp(col("date_next_purchase")) - unix_timestamp(col("timestamp"))) / 3600)
    spark_df = spark_df.withColumn("futur_purchase", when(col("hours_between") <= 1, 1).otherwise(0))

    # Group all purchase of a customer
    customer_group = spark_df.groupBy("customer_id").agg(sum("total_amount").alias("total_spend"),
                                                         sum("quantity").alias("total_quantity"),
                                                         count("*").alias("nb_transaction"),
                                                         max("futur_purchase").alias("futur_purchase"),
                                                         first("customer_type").alias("customer_type"))

    # More weight on probability 0 because 2 time less data when 0
    customer_group = customer_group.withColumn("class_weight", when(col("futur_purchase") == 1, 1.0).otherwise(2.0))

    # We encode customer type string to number
    logger.debug("Encoding and assembling the features")

    string_type = StringIndexer(inputCol="customer_type", outputCol="index_customer_type")
    customer_group = string_type.fit(customer_group).transform(customer_group)

    # We assemble the features
    vector = VectorAssembler(inputCols=["total_spend", "total_quantity", "nb_transaction", "index_customer_type"],
                             outputCol="features")
    customer_group = vector.transform(customer_group)

    # We split the data for train and test the model
    train, test = customer_group.randomSplit([0.8, 0.2], seed=123)

    # We will use randomforestclassifier to predict future purchase
    logger.debug("Create and train the model Random Forest Classifier")

    model = RandomForestClassifier(featuresCol="features", labelCol="futur_purchase", weightCol="class_weight")
    model_train = model.fit(train)

    logger.debug("Evaluate the model Random Forest Classifier")

    # We will evaluate the model
    eva = BinaryClassificationEvaluator(labelCol="futur_purchase")
    acc = eva.evaluate(model_train.transform(test))
    logger.debug(f"Accuracy : {acc}")

    logger.debug(customer_group.groupBy("futur_purchase").count().show())

    # PLOT
    logger.debug("Plot Random Forest Classifier")

    predi = model_train.transform(test)
    predi_df = predi.select("customer_id", "futur_purchase", col("probability").alias("predicted_probability"),
                            col("prediction").alias("predicted_label"))
    # Convert to pandas
    predi_pandas_df = predi_df.limit(100).toPandas()
    predi_pandas_df["predicted_probability"] = predi_pandas_df["predicted_probability"].apply(lambda x: x[1])
    # difference between predicted and true values
    predi_pandas_df["difference"] = abs(predi_pandas_df["predicted_label"] - predi_pandas_df["futur_purchase"])
    plt.figure(figsize=(12, 8))
    plt.scatter(range(len(predi_pandas_df)), predi_pandas_df["futur_purchase"], color="blue", marker="o",
                label="True Values", s=50)
    plt.scatter(range(len(predi_pandas_df)), predi_pandas_df["predicted_label"], color="orange", marker="x",
                label="Predicted Values", s=50)
    plt.plot(range(len(predi_pandas_df)), predi_pandas_df["difference"], linestyle="--", color="red", alpha=0.5,
             label="Difference")
    plt.title("True vs Predicted Purchase Values", fontsize=16)
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Purchase Prediction", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig( f"{dir_path}/predict_purchase_next_hour.png", dpi=300, bbox_inches="tight")
    plt.show()

    ###################################################################
    # Do forecast of total_amount purchase of each people. We want
    # to know how many they will spend on the next purchase.

    # The forecast have not good result because of ressource limitation
    # It make me 1h to train with 20 epochs,
    ###################################################################

    logger.debug("Begin dataset for LSTM")

    # group transaction by customer
    window_size = 5
    window2 = Window.partitionBy("customer_id").orderBy("timestamp")
    # We take the last 5 total_amount value
    lstm_data = spark_df.withColumn("sequence",
                                    collect_list("total_amount").over(window2.rowsBetween(-window_size + 1, 0)))

    # We keep only people with 5 total_amount to do the dataset
    lstm_data = lstm_data.filter(col("sequence").isNotNull() & (size(col("sequence")) == window_size))
    logger.debug(lstm_data.select("customer_id", "sequence", "total_amount").show(10, truncate=False))

    # We convert the spark dataframe to numpy for be able to use it on pytorch
    def spark_to_numpy(df, feature_col, target_col):
        features = np.array(df.select(feature_col).rdd.map(lambda x: x[0]).collect())
        targets = np.array(df.select(target_col).rdd.map(lambda x: x[0]).collect())
        return features, targets

    features, targets = spark_to_numpy(lstm_data, "sequence", "total_amount")
    logger.debug(f"Features shape: {features.shape}, Targets shape: {targets.shape}")

    logger.debug("Normalize datas for LSTM")

    # We normalize the datas
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    features_scaled = feature_scaler.fit_transform(features.reshape(-1, window_size))
    targets_scaled = target_scaler.fit_transform(targets.reshape(-1, 1))
    features_scaled = features_scaled.reshape(-1, window_size, 1)
    targets_scaled = targets_scaled[0::25000]
    features_scaled = features_scaled[0::25000]

    logger.debug(f"Features scaled: min={features_scaled.min()}, max={features_scaled.max()}")
    logger.debug(f"Targets scaled: min={targets_scaled.min()}, max={targets_scaled.max()}")

    logger.debug("Time Series Dataset Class")

    # Do a dataset class for pytorch and easy batch
    class TimeSeriesDataset(Dataset):
        def __init__(self, features, targets):
            self.features = torch.tensor(features, dtype=torch.float32)
            self.targets = torch.tensor(targets, dtype=torch.float32)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.targets[idx]

    # Create the LSTM model with one hidden layer and batchnorm
    logger.debug("LSTMModel Class")

    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.bn = nn.BatchNorm1d(hidden_dim)
            self._initialize_weights()

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.bn(out[:, -1, :])
            out = self.fc(out)
            return out

        def _initialize_weights(self):
            for name, param in self.named_parameters():
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    model = LSTMModel(1, 50, 1, 2)

    logger.debug("Training the LSTM")

    # TTraining of the model
    dataset = TimeSeriesDataset(features_scaled, targets_scaled)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = Adam(model.parameters(), lr=0.0001)
    epochs = 500
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step(epoch_loss / len(train_loader))
        logger.debug(scheduler.get_last_lr())
        logger.debug(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    # Forecast the 10 last sequences to visualize our result
    logger.debug("Forecast for plot LSTM")

    def forecast(model, data):
        model.eval()
        with torch.no_grad():
            data = torch.tensor(data, dtype=torch.float32)
            predictions = model(data)
        return predictions.squeeze().numpy()

    test_data = features_scaled[-10:]
    forecasted_values_scaled = forecast(model, test_data)
    forecasted_values = target_scaler.inverse_transform(forecasted_values_scaled.reshape(-1, 1))
    logger.debug(f"Forecasted Values: {forecasted_values}")

    forecast_df = spark.createDataFrame(
        [(float(val),) for val in forecasted_values], schema=["predicted_total_amount"]
    )
    forecast_df.show()

    # PLOT
    logger.debug("Plot LSTM")

    actual = targets[-10:]
    plt.scatter(range(len(actual)), actual, label="Actual", marker="o")
    plt.scatter(range(len(forecasted_values)), forecasted_values, label="Forecasted", marker="x")
    plt.title("True value vs Forecasted value")
    plt.xlabel("The n-ieme sequence")
    plt.ylabel("Total amount")
    plt.legend()
    plt.savefig(f"{dir_path}/forecast_predict.png", dpi=300, bbox_inches="tight")
    plt.show()
