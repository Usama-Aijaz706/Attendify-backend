# Attendance API Documentation

## GET /attendance/{roll_no}

Retrieve all attendance records for a specific student by their roll number.

### Endpoint
```
GET /attendance/{roll_no}
```

### Path Parameters
- `roll_no` (string, required): The roll number of the student

### Example Request
```bash
curl -X GET "http://your-server:8000/attendance/10148"
```

### Response Format

#### Success Response (200 OK)
```json
{
  "status": "success",
  "roll_no": "10148",
  "attendance_records": [
    {
      "_id": "507f1f77bcf86cd799439011",
      "student_id": "507f1f77bcf86cd799439012",
      "roll_no": "10148",
      "name": "Usama Aijaz",
      "class_name": "BSCS 8th",
      "section": "B",
      "teacher_name": "Miss Maria Khattak",
      "date": "2025-01-16",
      "time": "20:18:03",
      "status": "Present",
      "subject_name": "Mathematics"
    }
  ],
  "total_records": 1
}
```

#### Empty Response (200 OK)
When no attendance records are found for the given roll number:
```json
{
  "status": "success",
  "roll_no": "99999",
  "attendance_records": [],
  "total_records": 0
}
```

#### Error Response (500 Internal Server Error)
```json
{
  "detail": "An error occurred while fetching attendance records: [error message]"
}
```

### Features
- ✅ Queries MongoDB `attendance_records` collection
- ✅ Returns all records matching the roll number (as string)
- ✅ Handles ObjectId serialization (converts to string)
- ✅ Returns empty array if no records found
- ✅ Includes total record count
- ✅ Proper error handling
- ✅ Visible in FastAPI docs (`/docs`)

### Usage Examples

#### Python with requests
```python
import requests

# Get attendance for roll number 10148
response = requests.get("http://your-server:8000/attendance/10148")
data = response.json()

if data["status"] == "success":
    records = data["attendance_records"]
    print(f"Found {data['total_records']} attendance records")
    for record in records:
        print(f"Date: {record['date']}, Subject: {record['subject_name']}, Status: {record['status']}")
```

#### JavaScript/Fetch
```javascript
fetch('http://your-server:8000/attendance/10148')
  .then(response => response.json())
  .then(data => {
    if (data.status === 'success') {
      console.log(`Found ${data.total_records} attendance records`);
      data.attendance_records.forEach(record => {
        console.log(`Date: ${record.date}, Subject: ${record.subject_name}, Status: ${record.status}`);
      });
    }
  });
```

### Database Schema
The endpoint queries the `attendance_records` collection with the following document structure:
```json
{
  "_id": "ObjectId('...')",
  "student_id": "ObjectId('...')",
  "roll_no": "10148",
  "name": "Usama Aijaz",
  "class_name": "BSCS 8th",
  "section": "B",
  "teacher_name": "Miss Maria Khattak",
  "date": "2025-01-16",
  "time": "20:18:03",
  "status": "Present",
  "subject_name": "Mathematics"
}
```

### Testing
Run the test script to verify the endpoint:
```bash
python test_attendance_endpoint.py
```

Make sure to:
1. Update the `BASE_URL` in the test script if your server is not running on localhost:8000
2. Update the `TEST_ROLL_NO` to use a roll number that exists in your database
3. Ensure your FastAPI server is running before testing 