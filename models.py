from typing import List, Optional
from datetime import datetime
from beanie import Document
from pydantic import Field, BaseModel
import uuid

class FaceEmbedding(BaseModel):
    embedding_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vector: List[float]

class Student(Document):
    roll_no: str = Field(unique=True, index=True)
    name: str
    class_name: str
    section: str
    face_embeddings: List[FaceEmbedding] = []

    class Settings:
        name = "students" # MongoDB collection name

class AttendanceRecord(Document):
    student_id: str  # MongoDB ObjectId string of the student
    roll_no: str
    name: str
    class_name: str
    section: str
    teacher_name: str
    date: str # Store as string for simplicity with current date format, e.g., "YYYY-MM-DD"
    time: str # Store as string, e.g., "HH:MM:SS"
    status: str = "Present"
    subject_name: Optional[str] = None  # New field for subject, now optional
    class_time: Optional[str] = None  # New field for class time, optional

    class Settings:
        name = "attendance_records" # MongoDB collection name 