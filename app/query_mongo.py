from pymongo import MongoClient

# MongoDB connection
client = MongoClient("mongodb://mongodb:27017/")
db = client["customer_data_db"]
collection = db["processed_data"]

top_customers = collection.find().sort("total_spend", -1).limit(10)
for customer in top_customers:
    print(customer)

client.close()
