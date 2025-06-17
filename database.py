from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from typing import List, Union
from models import Student, AttendanceRecord

async def initiate_database(mongo_uri: str, database_name: str):
    client = AsyncIOMotorClient(mongo_uri)
    await init_beanie(database=client[database_name], document_models=[Student, AttendanceRecord]) 