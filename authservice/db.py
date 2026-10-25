from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv(override=True)

MONGO_URI = os.getenv("mongo_uri")
DB_NAME =  "prompteval"

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]


collection1=db["auth"]
evaluation=db["evaluations"]