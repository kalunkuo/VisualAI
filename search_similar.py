from pymongo import MongoClient
import json
import os

# Connect to Atlas (replace with your URI)
uri = os.getenv("MONGO_URI")
client = MongoClient(uri)
db = client["wound_ai_db"]
collection = db["embeddings"]

# Load an example embedding from file (or generate one)
with open("embeddings.json", "r") as f:
    data = json.load(f)

# Pick an embedding to query against
query_vector = data[0]["embedding"]

# Run $vectorSearch
results = db.command({
    "aggregate": "embeddings",
    "pipeline": [
        {
            "$vectorSearch": {
                "index": "vector_index_2", # or your custom index name
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 100,
                "limit": 5
            }
        }
    ],
    "cursor": {}
})

# Show results
print("\n🔍 Top 5 Similar Wound Images:")
for doc in results["cursor"]["firstBatch"]:
    print(f"{doc['filename']} ({doc['label']})")


doc = collection.find_one()
print(len(doc['embedding']))  # Should be 2048

print("Number of embeddings in MongoDB:", collection.count_documents({"embedding": {"$exists": True}}))
