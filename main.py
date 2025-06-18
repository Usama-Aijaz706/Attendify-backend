from dotenv import load_dotenv
import pandas as pd
import io

load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import face_recognition
import uvicorn
from database import initiate_database
from models import Student, FaceEmbedding, AttendanceRecord
from typing import List, Optional
import os
from datetime import datetime
import uuid
from face_service import preprocess_image_for_detection, extract_face_embeddings_from_image, get_face_locations_and_embeddings
from starlette.concurrency import run_in_threadpool

app = FastAPI()

# Environment variables for MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "attendify_db")

@app.on_event("startup")
async def start_database():
    print(f"Attempting to connect to MONGO_URI: {MONGO_URI}")
    print(f"Using DATABASE_NAME: {DATABASE_NAME}")
    await initiate_database(MONGO_URI, DATABASE_NAME)
    print("MongoDB connection initiated.")

@app.on_event("shutdown")
async def shutdown_database():
    # Beanie handles client closing, but explicit client shutdown might be needed for some use cases.
    # For motor, client.close() is usually handled by Beanie's lifecycle if using Document.find_one/save etc.
    # No direct motor client.close() needed if Beanie manages it.
    print("MongoDB connection closed.")

# Allow CORS for your React Native app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    try:
        # Read the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "Invalid image format"}

        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run face recognition
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        return {
            "status": "success",
            "faces_detected": len(face_locations),
            "face_locations": face_locations
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def read_root():
    return {"status": "success", "message": "Face Recognition API is running"}

# --- New Admin Endpoints ---

@app.post("/admin/students/register")
async def register_student(
    roll_no: str = Form(...),
    name: str = Form(...),
    class_name: str = Form(...),
    section: str = Form(...),
    images: List[UploadFile] = File(..., description="One or more student face images")
):
    extracted_embeddings = []
    for img_file in images:
        contents = await img_file.read()
        rgb_img = await run_in_threadpool(preprocess_image_for_detection, contents)

        if rgb_img is None:
            raise HTTPException(status_code=400, detail=f"Invalid image format for {img_file.filename}")
        
        face_encodings = await run_in_threadpool(extract_face_embeddings_from_image, rgb_img)

        if not face_encodings:
            raise HTTPException(status_code=400, detail=f"No clear, detectible human faces found in {img_file.filename} (min size 80px, aspect ratio 1:1.5). Please upload a clearer image.")

        extracted_embeddings.append(FaceEmbedding(vector=face_encodings[0].tolist()))

    # Check if student already exists by roll_no and name
    existing_student = await Student.find_one({"roll_no": roll_no, "name": name})
    if existing_student:
        # If student exists, REPLACE their embeddings with the new ones
        existing_student.face_embeddings = extracted_embeddings
        await existing_student.save()
        return {"status": "success", "message": f"Student {name} (Roll No: {roll_no}) embeddings updated.", "student_id": str(existing_student.id)}

    else:
        # Create new student
        new_student = Student(
            roll_no=roll_no,
            name=name,
            class_name=class_name,
            section=section,
            face_embeddings=extracted_embeddings
        )
        await new_student.insert()

        return {"status": "success", "message": f"Student {name} (Roll No: {roll_no}) registered and embeddings stored.", "student_id": str(new_student.id)}

@app.post("/admin/students/bulk_register_metadata")
async def bulk_register_metadata(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        df = None
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a CSV or Excel file.")

        required_columns = {'roll_no', 'name', 'class_name', 'section'}
        # Normalize column names to check for existence
        df.columns = df.columns.str.lower().str.replace(' ', '_')

        if not required_columns.issubset(df.columns):
            missing_cols = required_columns - set(df.columns)
            raise HTTPException(status_code=400, detail=f"Missing required columns in file: {', '.join(missing_cols)}. Required: roll_no, name, class_name, section.")

        successful_uploads = 0
        failed_uploads = []

        for index, row in df.iterrows():
            roll_no = str(row['roll_no']).strip()
            name = str(row['name']).strip()
            class_name = str(row['class_name']).strip()
            section = str(row['section']).strip()

            # Skip if any crucial field is empty
            if not roll_no or not name or not class_name or not section:
                failed_uploads.append({"row": index + 2, "message": "Missing required data (roll_no, name, class_name, or section)"})
                continue

            try:
                # Check if student already exists by roll_no and name
                existing_student = await Student.find_one({"roll_no": roll_no, "name": name})
                if existing_student:
                    # Update existing student's metadata
                    existing_student.class_name = class_name
                    existing_student.section = section
                    # Note: embeddings are NOT updated here. They should be uploaded via /admin/students/register
                    await existing_student.save()
                    successful_uploads += 1
                else:
                    # Create new student (without embeddings)
                    new_student = Student(
                        roll_no=roll_no,
                        name=name,
                        class_name=class_name,
                        section=section,
                        face_embeddings=[] # Initialize as empty, images added separately
                    )
                    await new_student.insert()
                    successful_uploads += 1

            except Exception as e:
                failed_uploads.append({"row": index + 2, "message": f"Database error: {str(e)}"})

        return {
            "status": "success",
            "message": f"Bulk metadata upload completed. {successful_uploads} students processed.",
            "successful_count": successful_uploads,
            "failed_records": failed_uploads
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during file processing: {str(e)}")

@app.get("/admin/students")
async def get_all_students():
    """Fetches all student records including their face embeddings."""
    students = await Student.find_all().to_list()
    # Convert Beanie documents to dictionaries for JSON serialization
    # Also convert numpy arrays in embeddings to lists if they somehow become numpy types
    students_data = []
    for student in students:
        student_dict = student.dict()
        # Ensure embeddings vectors are lists of floats, not numpy arrays
        if 'face_embeddings' in student_dict:
            student_dict['face_embeddings'] = [
                {'embedding_id': emb['embedding_id'], 'vector': emb['vector']}
                for emb in student_dict['face_embeddings']
            ]
        students_data.append(student_dict)
    return {"status": "success", "students": students_data}

# --- New Attendance Endpoints ---

@app.post("/attend/process_frame")
async def process_attendance_frame(
    file: UploadFile = File(...),
    class_id: str = Form(...),
    teacher_name: str = Form(...),
    subject_name: str = Form(...),  # New field for subject
    date: Optional[str] = Form(None) # Allow date to be sent or determined by backend
):
    print(f"Received frame for attendance in class {class_id} by {teacher_name}")

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    contents = await file.read()
    rgb_img = await run_in_threadpool(preprocess_image_for_detection, contents)

    if rgb_img is None:
        raise HTTPException(status_code=400, detail="Invalid image format for video frame.")
    
    face_locations, face_encodings = await run_in_threadpool(get_face_locations_and_embeddings, rgb_img)

    recognized_students = []
    matched_students_ids = set()

    if not face_encodings:
        return {"status": "success", "recognized_students": [], "message": "No clear, detectible human faces found in the frame."}

    # Fetch all student embeddings from the database
    all_students = await Student.find_all().to_list()
    known_face_encodings = []
    known_student_data = [] # To store details of known students for easy lookup

    for student in all_students:
        for embedding_obj in student.face_embeddings:
            known_face_encodings.append(np.array(embedding_obj.vector))
            known_student_data.append({
                "student_id": str(student.id),
                "roll_no": student.roll_no,
                "name": student.name,
                "class_name": student.class_name,
                "section": student.section
            })

    # Normalize fields for robust matching
    norm_class_id = class_id.strip().lower()
    norm_teacher_name = teacher_name.strip().lower()
    norm_subject_name = subject_name.strip().lower()
    # We'll also normalize section and student fields

    for i, face_encoding in enumerate(face_encodings):
        # Get the corresponding face_location for the current face_encoding
        current_face_location = face_locations[i]

        if known_face_encodings:
            matches = await run_in_threadpool(face_recognition.compare_faces, known_face_encodings, face_encoding, tolerance=0.6)
            face_distances = await run_in_threadpool(face_recognition.face_distance, known_face_encodings, face_encoding)
            
            best_match_index = -1
            if True in matches:
                # Find the best match (lowest distance among matches)
                matched_indices = [i for i, x in enumerate(matches) if x]
                best_match_index = matched_indices[np.argmin(face_distances[matched_indices])]
            
            if best_match_index != -1:
                matched_student = known_student_data[best_match_index]
                student_obj_id = matched_student["student_id"]
                norm_section = matched_student["section"].strip().lower()
                # Check if attendance already marked for this subject, student, and day
                attendance_query = {
                    "student_id": student_obj_id,
                    "class_name": norm_class_id,
                    "date": date,
                    "teacher_name": norm_teacher_name,
                    "section": norm_section,
                    "subject_name": norm_subject_name
                }
                print(f"Attendance query: {attendance_query}")
                existing_attendance = await AttendanceRecord.find_one(attendance_query)
                print(f"Existing attendance found: {existing_attendance is not None}")
                student_response_data = {
                    **matched_student,
                    "face_location": list(current_face_location) # Add face location to the response
                }
                if not existing_attendance:
                    attendance_record = AttendanceRecord(
                        student_id=student_obj_id,
                        roll_no=matched_student["roll_no"],
                        name=matched_student["name"],
                        class_name=norm_class_id,
                        section=norm_section,
                        teacher_name=norm_teacher_name,
                        date=date,
                        time=current_time,
                        status="Present",
                        subject_name=norm_subject_name
                    )
                    await attendance_record.insert()
                    recognized_students.append({**student_response_data, "status": "Present"})
                else:
                    recognized_students.append({**student_response_data, "status": "Already Present"})
                matched_students_ids.add(student_obj_id)

    return {"status": "success", "recognized_students": recognized_students}

if __name__ == "__main__":
    # For local testing, ensure MONGO_URI and DATABASE_NAME are set in your .env file or environment
    uvicorn.run(app, host="0.0.0.0", port=8000) 