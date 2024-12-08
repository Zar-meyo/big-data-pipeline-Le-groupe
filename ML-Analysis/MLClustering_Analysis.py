import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, count, collect_set, expr, hour, dayofweek, month
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, CountVectorizer, PCA
from pyspark.ml.clustering import KMeans
import matplotlib.pyplot as plt

#df = pd.read_csv('../app/ecommerce_data_with_trends.csv')
#print(df.columns)

spark = SparkSession.builder.appName("customer_segmentation").getOrCreate()
df = spark.read.csv('../app/ecommerce_data_with_trends.csv', header=True, inferSchema=True)
df.show(5)
#df.printSchema()

################################################################################
# We will identify customer group using K-means and
# features of the dataset (city, customer_type, quantity
# and total_amount)
################################################################################

#We encode customer type string and city string to number
customer_type = StringIndexer(inputCol="customer_type", outputCol="customer_type_index")
city_type = StringIndexer(inputCol="city", outputCol="city_index")
customer_group_data = customer_type.fit(df).transform(df)
customer_group_data = city_type.fit(customer_group_data).transform(customer_group_data)

#We assemble the featurres
vector = VectorAssembler(inputCols=["city_index", "customer_type_index", "quantity", "total_amount"], outputCol="feature")
customer_group_data = vector.transform(customer_group_data)

#Normalize data
scaler = StandardScaler(inputCol="feature", outputCol="scale_feature")
customer_group_data = scaler.fit(customer_group_data).transform(customer_group_data)

#We do K-means silhouette_score evaluation to identify the best nb of clusters
"""
silhouette_score = []
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='scale_feature', metricName='silhouette', distanceMeasure='squaredEuclidean')
for i in range(3, 20):
    kmeans = KMeans(featuresCol='scale_feature', k=i)
    model = kmeans.fit(customer_group_data)
    predictions = model.transform(customer_group_data)
    score = evaluator.evaluate(predictions)
    silhouette_score.append(score)
    print('Silhouette Score for k =', i, 'is', score)
"""

#Best result for silhouette score was with k=5 (0.658)
kmeans = KMeans(featuresCol="scale_feature", k=5, seed=42)
model = kmeans.fit(customer_group_data)

predictions = model.transform(customer_group_data)

#Convert to Pandas for plot
plot_data = predictions.select("city_index", "customer_type_index", "quantity", "total_amount", "prediction").toPandas()
#Just take 1 out of 40 points (if we take all it's too much)
plot_data = plot_data.iloc[::40, :]
#We will take city_index, customer_type_index and total_amount information to plot
#This will do cluster of customer depending on their type, city and spent
plot_data["x"] = plot_data["city_index"]
plot_data["y"] = plot_data["customer_type_index"]
plot_data["z"] = plot_data["total_amount"]
#PLOT
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
for cluster_id in sorted(plot_data["prediction"].unique()):
    cluster_data = plot_data[plot_data["prediction"] == cluster_id]
    ax.scatter(cluster_data["x"], cluster_data["y"], cluster_data["z"], label=f"Cluster {cluster_id}", alpha=0.6)
ax.set_title("Cluster of customer groups identified by K-Means")
ax.set_xlabel("City")
ax.set_ylabel("Customer_type")
ax.set_zlabel("Total_amount")
ax.legend()
plt.tight_layout()
plt.savefig("Plot_result/customer_depending_city_amount_type_features.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


################################################################################
# We will identify relation between customer and their similar purchase
# (using category and product_name)
################################################################################

#We encode category string and product_name string to number
category_type = StringIndexer(inputCol="category", outputCol="category_index")
product_name_type = StringIndexer(inputCol="product_name", outputCol="product_name_index")
customer_group2_data = category_type.fit(df).transform(df)
customer_group2_data = product_name_type.fit(customer_group2_data).transform(customer_group2_data)

#Group all buy of a customer
customer_buy = customer_group2_data.groupBy("customer_id").agg(
    collect_set("category_index").alias("collect_categories"),
    collect_set("product_name_index").alias("collect_products_name")
)
#Cast arrays to string
customer_buy = customer_buy.withColumn(
    "collect_categories", expr("transform(collect_categories, x -> cast(x as string))")).withColumn(
    "collect_products_name", expr("transform(collect_products_name, x -> cast(x as string))")
)
vector_category = CountVectorizer(inputCol="collect_categories", outputCol="vector_categories")
vector_product_name = CountVectorizer(inputCol="collect_products_name", outputCol="vector_products_name")
customer_buy = vector_category.fit(customer_buy).transform(customer_buy)
customer_buy = vector_product_name.fit(customer_buy).transform(customer_buy)

#We assemble the features
vector2 = VectorAssembler(inputCols=["vector_categories", "vector_products_name"], outputCol="features")
customer_buy = vector2.transform(customer_buy)

# Do PCA for dimensionality reduction
# I choose PCA of 3 components to be able to plot all the component of the
# PCA, even if we loose informations with that
pca = PCA(k=3, inputCol="features", outputCol="pca_features")
customer_buy = pca.fit(customer_buy).transform(customer_buy)

#Best : k = 4 (0.48)
"""
silhouette_score2 = []
evaluator2 = ClusteringEvaluator(predictionCol='prediction', featuresCol='pca_features', metricName='silhouette', distanceMeasure='squaredEuclidean')
for i in range(3, 20):
    kmeans = KMeans(featuresCol='pca_features', k=i)
    model = kmeans.fit(customer_buy)
    predictions = model.transform(customer_buy)
    score = evaluator2.evaluate(predictions)
    silhouette_score2.append(score)
    print('Silhouette Score for k =', i, 'is', score)
"""

kmeans2 = KMeans(featuresCol="pca_features", k=4, seed=42)
model2 = kmeans2.fit(customer_buy)

#PLOT
#We will see the 4 clusters using the informations of the
#three composents of our PCA
predictions = model2.transform(customer_buy)
plot_data = predictions.select("customer_id", "pca_features", "prediction").toPandas()
plot_data["x"] = plot_data["pca_features"].apply(lambda v: v[0])
plot_data["y"] = plot_data["pca_features"].apply(lambda v: v[1])
plot_data["z"] = plot_data["pca_features"].apply(lambda v: v[2])
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
for c in sorted(plot_data["prediction"].unique()):
    cluster_data = plot_data[plot_data["prediction"] == c]
    ax.scatter(cluster_data["x"], cluster_data["y"], cluster_data["z"], label=f"Cluster {c}", alpha=0.6)
ax.set_title("Customer clusters of their similar purchase")
ax.set_xlabel("PCA first component")
ax.set_ylabel("PCA second component")
ax.set_zlabel("PCA third component")
ax.legend()
plt.savefig("Plot_result/customer_similar_buy.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()

#Now, we will find top 10 feature which contribute the most and worst
#for the 3 components of our PCA

#He don't find .vocabulary so i have to drop the columns and
#redo all i don't know why
customer_buy = customer_buy.drop("pca_features")
pca_model = pca.fit(customer_buy)
pca_loadings = pca_model.pc.toArray()
explained_variance = pca_model.explainedVariance.toArray()
customer_buy = customer_buy.drop("vector_products_name")
customer_buy = customer_buy.drop("vector_categories")
cv_category_model = vector_category.fit(customer_buy)
cv_product_model = vector_product_name.fit(customer_buy)
category_vocab = cv_category_model.vocabulary
product_vocab = cv_product_model.vocabulary
all_features = category_vocab + product_vocab
n_components = pca_loadings.shape[1]

#Create a dataframe for PCA to use it for top 10
loading_df = pd.DataFrame(pca_loadings, index=all_features, columns=[f"PC{i+1}" for i in range(n_components)])
category_indexer_model = category_type.fit(df)
product_indexer_model = product_name_type.fit(df)

#Decode category and product type to have string
category_labels = category_indexer_model.labels
product_labels = product_indexer_model.labels
decoded_features = []
for feature in category_vocab:
    feature_index = int(float(feature))
    if feature_index < len(category_labels):
        decoded_features.append(f"Category: {category_labels[feature_index]}")
for feature in product_vocab:
    feature_index = int(float(feature))
    if feature_index < len(product_labels):
        decoded_features.append(f"Product: {product_labels[feature_index]}")
decoded_df = loading_df.copy()
decoded_df.index = decoded_features

#PLOT for each PCA component
for pc in ["PC1", "PC2", "PC3"]:
    # Get the best and worst top 10 features for each component
    top_10_features = decoded_df[pc].sort_values(ascending=False).head(10)
    bottom_10_features = decoded_df[pc].sort_values(ascending=True).head(10)
    top_bottom_features = pd.concat([top_10_features, bottom_10_features])

    #PLOT
    plt.figure(figsize=(12, 8))
    top_bottom_features.plot(kind="bar", color=["green" if v > 0 else "red" for v in top_bottom_features.values])
    plt.title(f"Top 10 best and worst feature to {pc}")
    plt.xlabel("Features")
    plt.ylabel("Contribution")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = f"Plot_result/top_bottom_{pc}_contributions_decoded_original.png"
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


################################################################################
# We will identify purchase time to analyze hours, days and months purchase
# trends of customer (to for exemple target customer for month ads campaigns
################################################################################

#Get temporal data to have columns of days, hours and months
time_data = df.withColumn("hour", hour("timestamp")).withColumn("day_of_week", dayofweek("timestamp")).withColumn("month", month("timestamp"))

#We assemble the time features
vector3 = VectorAssembler(inputCols=["hour", "day_of_week", "month"], outputCol="time_features")
time_data = vector3.transform(time_data)

#Normalize datas
scaler2 = StandardScaler(inputCol="time_features", outputCol="time_features_scaled")
time_data = scaler2.fit(time_data).transform(time_data)

#Best : k = 8
"""
silhouette_score3 = []
evaluator3 = ClusteringEvaluator(predictionCol='prediction', featuresCol='time_features_scaled', metricName='silhouette', distanceMeasure='squaredEuclidean')
for i in range(3, 20):
    kmeans = KMeans(featuresCol='time_features_scaled', k=i)
    model = kmeans.fit(time_data)
    predictions = model.transform(time_data)
    score = evaluator3.evaluate(predictions)
    silhouette_score3.append(score)
    print('Silhouette Score for k =', i, 'is', score)
"""

#Do Kmeans with k=8
kmeans3 = KMeans(featuresCol="time_features_scaled", k=8, seed=42)
model3 = kmeans3.fit(time_data)
time_clusters = model3.transform(time_data)

#PLOT
plot_data = time_clusters.select("hour", "day_of_week", "month", "prediction").toPandas()
plot_data = plot_data.iloc[::40, :]
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
for c in sorted(plot_data["prediction"].unique()):
    cluster_data = plot_data[plot_data["prediction"] == c]
    ax.scatter(cluster_data["hour"], cluster_data["day_of_week"], cluster_data["month"], label=f"Cluster {c}", alpha=0.6)
ax.set_title("Customer cluster of their purchases date trends")
ax.set_xlabel("Hour")
ax.set_ylabel("Day of the week")
ax.set_zlabel("Month")
ax.legend()
plt.tight_layout()
plt.savefig("Plot_result/time_clusters.png", dpi=300, bbox_inches="tight")
plt.show()


spark.stop()