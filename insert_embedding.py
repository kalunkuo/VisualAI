from pymongo import MongoClient
import random
import os
import json

#$env:MONGO_URI = "mongodb+srv://your_user:your_password@your_cluster.mongodb.net"
uri = os.getenv("MONGO_URI")
client = MongoClient(uri)
db = client["wound_ai_db"]
collection = db["embeddings"]

# Load your local embeddings JSON
with open("embeddings.json", "r") as f:
    data = json.load(f)

# Insert into MongoDB
collection.insert_many(data)
print(f"✅ Inserted {len(data)} documents into MongoDB Atlas.")