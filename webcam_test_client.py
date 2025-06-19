import cv2
import requests
import numpy as np
import time
import io
import face_recognition
import threading
from queue import Queue
import uuid # For generating unique local IDs

# --- Configuration --- #
BACKEND_BASE_URL = "http://localhost:8000"
ATTENDANCE_ENDPOINT = f"{BACKEND_BASE_URL}/attend/process_frame"
GET_STUDENTS_ENDPOINT = f"{BACKEND_BASE_URL}/admin/students"

# NOTE: For this client, ensure CLASS_ID matches the class_name stored in your student data
CLASS_ID = "BSCS 8th" # Example class ID, ensure consistency with DB
TEACHER_NAME = "Miss Maria Khattak" # Example teacher name
SUBJECT_NAME = "Mathematics"  # Example subject name, change as needed

# --- Performance Settings --- #
FRAME_SKIP = 10  # Process every 10th frame for local detection (improved performance)
SEND_INTERVAL = 2.0  # seconds between sending frames to backend for new detections (increased for performance)
MIN_FACE_SIZE = 80
MAX_ASPECT_RATIO = 1.5
RECOGNITION_TOLERANCE = 0.6 # How strict face_recognition matching is
FACE_TRACKING_THRESHOLD = 0.5 # Max distance to consider a face the "same" across frames locally
UNRECOGNIZED_RETRY_INTERVAL = 5.0 # seconds before sending an unrecognized face to backend again

# --- Global variables --- #
frame_queue = Queue(maxsize=2)  # Queue for frames to be processed
sending_frames = False
last_frame_sent_time = 0
status_message = "Press 's' to start"
# recognized_students_display = [] # REMOVED: Merged into detected_faces_history for direct display
marked_student_ids_in_session = set() # Store student_ids already marked as Present

# Store all known student embeddings locally for faster client-side recognition
# Format: {student_id: {data: {name, roll_no, class_name, section, id}, embeddings: [np.array, ...]}}
known_students_data = {}

# Local tracking of detected faces in recent frames for ID assignment and status
# {local_face_id: {'last_seen': timestamp, 'last_location': (top, right, bottom, left), 'last_embedding': np.array,
#                 'status': 'unknown'/'known_unmarked'/'known_marked', 'backend_id': student_id if known,
#                 'display_name': str, 'display_status': str, 'last_sent_to_backend': timestamp}}
detected_faces_history = {}

# Lock for thread-safe access to global variables (if needed for complex scenarios, simplified for now)
# data_lock = threading.Lock()

def filter_local_face_locations(face_locations):
    filtered_locations = []
    for top, right, bottom, left in face_locations:
        width = right - left
        height = bottom - top
        if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
            continue
        if width > 0 and height > 0:
            aspect_ratio = max(width / height, height / width)
            if aspect_ratio > MAX_ASPECT_RATIO:
                continue
        filtered_locations.append((top, right, bottom, left))
    return filtered_locations

def get_all_students_from_backend():
    """Fetches all student data and embeddings from the backend.
       This is done once to populate local known_students_data.
    """
    print("Fetching all student data from backend...")
    try:
        response = requests.get(GET_STUDENTS_ENDPOINT)
        response.raise_for_status()
        students_data = response.json().get("students", [])
        
        temp_known_students_data = {}
        for student in students_data:
            student_id = student["id"]
            embeddings = [np.array(e["vector"]) for e in student.get("face_embeddings", [])]
            if embeddings:
                temp_known_students_data[student_id] = {
                    "data": {
                        "id": student_id, # Correctly store as 'id'
                        "roll_no": student["roll_no"],
                        "name": student["name"],
                        "class_name": student["class_name"],
                        "section": student["section"]
                    },
                    "embeddings": embeddings
                }
        print(f"Successfully loaded {len(temp_known_students_data)} students for local recognition.")
        return temp_known_students_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching students from backend: {e}")
        return {}

# NOTE: Need to add a backend endpoint to get all students (simple GET /admin/students)
# For now, let's assume such an endpoint exists.
# If not, you might need to add it to main.py later.

def process_frame_for_local_recognition(frame):
    global detected_faces_history
    current_time = time.time()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    filtered_face_locations = filter_local_face_locations(face_locations)

    current_frame_detected_local_ids = set() # Keep track of local_ids found in this frame

    if not filtered_face_locations:
        # If no faces detected, mark old entries as not seen, clean up
        for local_id in list(detected_faces_history.keys()): # Iterate over copy to allow modification
            detected_faces_history[local_id]['last_seen_actual'] = 0 # Mark as not seen
        # Clean up history more aggressively if no faces at all
        faces_to_remove = [local_id for local_id, info in detected_faces_history.items() if (current_time - info['last_seen_actual']) > 30] # Remove after 30 seconds of not seeing
        for local_id in faces_to_remove:
            del detected_faces_history[local_id]
        return [] # No faces to process or display
    
    face_encodings = face_recognition.face_encodings(rgb_frame, filtered_face_locations)
    
    for i, face_encoding in enumerate(face_encodings):
        current_face_location = filtered_face_locations[i]
        
        # Try to match with existing faces in history to maintain ID
        matched_local_id = None
        min_local_dist = float('inf')

        for local_id, history_info in detected_faces_history.items():
            if 'last_embedding' in history_info: # Only if history has an embedding
                local_dist = face_recognition.face_distance([history_info['last_embedding']], face_encoding)[0]
                if local_dist < min_local_dist and local_dist <= FACE_TRACKING_THRESHOLD:
                    min_local_dist = local_dist
                    matched_local_id = local_id

        if matched_local_id:
            # Update existing face in history
            local_id = matched_local_id
            detected_faces_history[local_id]['last_seen_actual'] = current_time # Actual last seen for cleanup
            detected_faces_history[local_id]['last_location'] = current_face_location
            detected_faces_history[local_id]['last_embedding'] = face_encoding
        else:
            # New face detected, assign new local ID
            local_id = str(uuid.uuid4()) # Use UUID for unique IDs
            detected_faces_history[local_id] = {
                'id': local_id,
                'last_seen_actual': current_time,
                'last_location': current_face_location,
                'last_embedding': face_encoding,
                'status': 'unknown', # Internal status: unknown/known_unmarked/known_marked
                'backend_id': None,
                'current_display_status': 'Scanning...', # What's shown on screen
                'last_sent_to_backend': 0 # Timestamp of last time this specific face was sent
            }
        
        current_frame_detected_local_ids.add(local_id)

        # Now, try to recognize against known students (local recognition)
        best_known_match = None
        min_known_distance = float('inf')

        for student_id, student_info in known_students_data.items():
            distances = face_recognition.face_distance(student_info["embeddings"], face_encoding)
            if len(distances) > 0:
                current_min_distance = np.min(distances)
                if current_min_distance < min_known_distance and current_min_distance <= RECOGNITION_TOLERANCE:
                    min_known_distance = current_min_distance
                    best_known_match = student_info["data"]
        
        if best_known_match:
            # Local recognition: This is a known student
            if best_known_match['id'] in marked_student_ids_in_session:
                # This student is known AND marked as present in this session (by backend)
                detected_faces_history[local_id]['status'] = 'known_marked'
                detected_faces_history[local_id]['backend_id'] = best_known_match['id']
                # If 'Present' was just shown, keep it for 2 seconds
                if 'present_shown_time' in detected_faces_history[local_id]:
                    if time.time() - detected_faces_history[local_id]['present_shown_time'] < 2:
                        detected_faces_history[local_id]['current_display_status'] = f"{best_known_match['name']} (Present)"
                    else:
                        detected_faces_history[local_id]['current_display_status'] = f"{best_known_match['name']} (Already Present)"
                else:
                    detected_faces_history[local_id]['current_display_status'] = f"{best_known_match['name']} (Already Present)"
            elif detected_faces_history[local_id]['status'] == 'unknown':
                # If previously unknown, update to known_unmarked and show as new
                detected_faces_history[local_id]['status'] = 'known_unmarked' # Known but not yet marked
                detected_faces_history[local_id]['backend_id'] = best_known_match['id']
                detected_faces_history[local_id]['current_display_status'] = f"{best_known_match['name']} (New)"
                # Remove present_shown_time if it exists
                detected_faces_history[local_id].pop('present_shown_time', None)
            # No else needed: if already known_unmarked and not yet marked by backend, just keep previous state.
        else:
            # Local recognition: This is an unknown face
            if detected_faces_history[local_id]['status'] not in ['known_unmarked', 'known_marked']:
                 detected_faces_history[local_id]['status'] = 'unknown'
                 detected_faces_history[local_id]['backend_id'] = None
                 detected_faces_history[local_id]['current_display_status'] = f"Face {local_id[:4]} (Unknown)"

    # Mark faces not detected in this frame as 'not_seen_recently' or clean up
    for local_id in list(detected_faces_history.keys()):
        if local_id not in current_frame_detected_local_ids:
            detected_faces_history[local_id]['last_seen_actual'] = 0 # Mark as not seen in this exact frame
    
    # Aggressively clean up truly old entries (not seen for a while)
    faces_to_remove = [local_id for local_id, info in detected_faces_history.items() if (current_time - info['last_seen_actual']) > 30] # Remove after 30 seconds of not seeing
    for local_id in faces_to_remove:
        del detected_faces_history[local_id]

    return [] # Return detailed info for drawing and sending decisions

def send_frame_to_backend(frame, class_id, teacher_name):
    """Send frame to backend in a separate thread"""
    try:
        is_success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if is_success:
            image_bytes = io.BytesIO(buffer).getvalue()
            files = {'file': ('webcam_frame.jpg', image_bytes, 'image/jpeg')}
            data = {
                'class_id': class_id,
                'teacher_name': teacher_name,
                'subject_name': SUBJECT_NAME,  # Send subject_name
                'date': time.strftime("%Y-%m-%d")
            }
            response = requests.post(ATTENDANCE_ENDPOINT, files=files, data=data, timeout=15.0)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Error sending frame to backend: {e}")
    return None

def backend_worker():
    """Worker thread for sending frames to backend and updating status."""
    global last_frame_sent_time, status_message, marked_student_ids_in_session, detected_faces_history
    
    while True:
        if not sending_frames:
            time.sleep(0.1)
            continue
            
        current_time = time.time()
        
        # Check if we should send a frame based on the SEND_INTERVAL
        if current_time - last_frame_sent_time < SEND_INTERVAL:
            time.sleep(0.05) # Small sleep to avoid busy-waiting too much
            continue
            
        # Get the latest frame from the queue
        if not frame_queue.empty():
            frame_to_send = frame_queue.get() # Get the latest frame
            
            # --- Improved Decision Logic for Sending to Backend ---
            # Send a frame if ANY face is detected in view
            faces_in_view = [face_info for face_info in detected_faces_history.values() if (current_time - face_info['last_seen_actual']) < 1]
            if not faces_in_view:
                status_message = "No faces in view."
                last_frame_sent_time = current_time # Update time to respect interval
                continue

            print("Sending frame to backend...")
            try:
                result = send_frame_to_backend(frame_to_send, CLASS_ID, TEACHER_NAME)
                print(f"Backend response: {result}")
            except Exception as e:
                print(f"Error sending frame to backend: {e}")
                status_message = f"Backend Error: {e}"
                last_frame_sent_time = current_time
                continue
            
            if result and result.get("status") == "success":
                recognized_students_from_backend = result.get("recognized_students", [])
                backend_recognized_map = {s['student_id']: s for s in recognized_students_from_backend}
                for local_id, face_info in list(detected_faces_history.items()):
                    if (current_time - face_info['last_seen_actual']) < 0.5:
                        backend_id = face_info['backend_id']
                        if backend_id and backend_id in backend_recognized_map:
                            backend_status_info = backend_recognized_map[backend_id]
                            if backend_status_info.get("status") == "Present":
                                face_info['status'] = 'known_marked'
                                face_info['present_shown_time'] = time.time()
                                face_info['current_display_status'] = 'Present'
                                marked_student_ids_in_session.add(backend_id)
                            elif backend_status_info.get("status") == "Already Present":
                                face_info['status'] = 'known_marked'
                                face_info['current_display_status'] = 'Already Present'
                                marked_student_ids_in_session.add(backend_id)
                            face_info['last_sent_to_backend'] = current_time
                        elif face_info['status'] == 'unknown':
                            face_info['last_sent_to_backend'] = current_time
                status_message = "Backend Response: OK"
            else:
                status_message = "Backend Error or Timeout"
            last_frame_sent_time = current_time
        else:
            time.sleep(0.05) # No frame in queue


# Start backend worker thread
backend_thread = threading.Thread(target=backend_worker, daemon=True)
backend_thread.start()

# --- OpenCV setup --- #
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 30)

# --- Subject selection prompt --- #
subject_input = input("Enter subject name for attendance (default: Mathematics): ").strip()
if subject_input:
    SUBJECT_NAME = subject_input
else:
    SUBJECT_NAME = "Mathematics"
print(f"Subject selected: {SUBJECT_NAME}")

print("Press 's' to start/stop sending frames to the backend.")
print("Press 'q' to quit.")

frame_count = 0
# local_detected_faces_info = [] # REMOVED: Merged into detected_faces_history for direct display

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_count += 1
    display_frame = frame.copy()

    # Perform local recognition on this frame (every FRAME_SKIP frames or if not sending for blue box)
    if frame_count % FRAME_SKIP == 0 or not sending_frames:
        process_frame_for_local_recognition(frame) # This updates detected_faces_history globally
    
    current_time = time.time()
    # --- Fix status message logic ---
    faces_in_view = any((current_time - face_info['last_seen_actual']) < 1 for face_info in detected_faces_history.values())
    if faces_in_view:
        status_message = ""
    else:
        status_message = "No faces in view."

    # Drawing logic based on global detected_faces_history
    for local_id, face_info in detected_faces_history.items():
        if (current_time - face_info['last_seen_actual']) < 1: # Only draw for recently seen faces
            top, right, bottom, left = face_info['last_location']
            text_to_display = f'{face_info["current_display_status"]}'
            box_color = (0, 0, 0) # Default transparent/black
            text_color = (255, 255, 255) # Default white

            if face_info['status'] == 'known_marked':
                # Show 'Present' for 1.5s, then 'Already Present'
                if 'present_shown_time' in face_info and (current_time - face_info['present_shown_time']) < 1.5:
                    text_to_display = 'Present'
                else:
                    text_to_display = 'Already Present'
                text_color = (255, 255, 0) # Yellow
                # No box drawn
            elif face_info['status'] == 'known_unmarked':
                box_color = (0, 255, 0) # Green for known, not marked
                text_color = (0, 255, 0) # Green
                cv2.rectangle(display_frame, (left, top), (right, bottom), box_color, 2)
            elif face_info['status'] == 'unknown':
                box_color = (0, 0, 255) # Red for unknown
                text_color = (0, 0, 255) # Red
                text_to_display = 'Unknown'
                cv2.rectangle(display_frame, (left, top), (right, bottom), box_color, 2)
            cv2.putText(display_frame, text_to_display, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # Display main application status message
    if status_message:
        cv2.putText(display_frame, status_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Add frame to queue if sending and queue is not full
    faces_in_view = any((current_time - face_info['last_seen_actual']) < 1 for face_info in detected_faces_history.values())
    if sending_frames and faces_in_view and frame_queue.qsize() < frame_queue.maxsize:
        frame_queue.put(frame.copy())

    cv2.imshow("Webcam Test Client", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        sending_frames = not sending_frames
        if sending_frames:
            status_message = "Sending frames..."
            marked_student_ids_in_session.clear()
            detected_faces_history.clear() # Clear tracking on session start/stop
            known_students_data = get_all_students_from_backend()
            if not known_students_data:
                status_message = "Error: Could not load student data for local recognition! Check backend."
                sending_frames = False
        else:
            status_message = "Stopped sending frames."
            marked_student_ids_in_session.clear()
            detected_faces_history.clear() # Clear tracking on session start/stop

cap.release()
cv2.destroyAllWindows() 