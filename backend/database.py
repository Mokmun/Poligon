from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ServerSelectionTimeoutError

client = AsyncIOMotorClient("mongodb+srv://tanadech2545:xT4vifelsy@cluster0.g1rcd.mongodb.net/")
db = client["3d-furniture"]
user = db["users"]
collection = db["collections"]


