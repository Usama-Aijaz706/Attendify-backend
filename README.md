# Face Recognition API

A FastAPI-based backend service for face recognition, designed to work with React Native applications.

## Features

- Face detection and recognition
- RESTful API endpoints
- CORS enabled for cross-origin requests
- Easy deployment on Render

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /`: Health check endpoint
- `POST /recognize`: Face recognition endpoint
  - Accepts image file upload
  - Returns face detection results

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Testing the API

You can test the API using tools like Postman or curl:

```bash
# Test the health check endpoint
curl http://localhost:8000/

# Test face recognition (replace path/to/image.jpg with your image)
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/recognize
```

## Error Handling

The API includes comprehensive error handling for:
- Invalid image formats
- Processing errors
- Server errors

## Security Considerations

- In production, update the CORS settings to only allow requests from your specific domains
- Consider implementing authentication for API endpoints
- Use HTTPS in production

## License

MIT License 