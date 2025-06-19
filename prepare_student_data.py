import os
import cv2
import face_recognition
import shutil
from pathlib import Path
import argparse
import json # Import json for metadata files
import numpy as np

# --- Configuration ---
# This is where your raw student image folders are located
# Example: 'C:\Users\sudai\OneDrive\Desktop\Student_Photos'
SOURCE_RAW_IMAGES_DIR = ""

# This is the directory where processed images and metadata will be stored for upload
TARGET_PROCESSED_IMAGES_DIR = "student_images"

# Number of clearest images to select per student
NUM_IMAGES_TO_SELECT = 3

# Minimum face size for local detection to be considered 'clear'
MIN_FACE_SIZE_FOR_SELECTION = 80 # pixels
MAX_ASPECT_RATIO_FOR_SELECTION = 1.5 # max width/height or height/width

def enhance_image(img):
    # Denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    # Histogram Equalization (on Y channel)
    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    y_eq = cv2.equalizeHist(y)
    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    img_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCrCb2BGR)
    return img_eq

def filter_and_score_face_locations(face_locations, image_shape):
    scored_locations = []
    img_height, img_width, _ = image_shape

    for top, right, bottom, left in face_locations:
        width = right - left
        height = bottom - top

        # Filter by minimum size
        if width < MIN_FACE_SIZE_FOR_SELECTION or height < MIN_FACE_SIZE_FOR_SELECTION:
            continue

        # Filter by aspect ratio
        if width > 0 and height > 0:
            aspect_ratio = max(width / height, height / width)
            if aspect_ratio > MAX_ASPECT_RATIO_FOR_SELECTION:
                continue
        
        # Calculate a simple score based on size and centrality
        area = width * height
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        
        # Distance from center of the image (normalize for different resolutions)
        dist_x_norm = abs(center_x - img_width / 2) / (img_width / 2)
        dist_y_norm = abs(center_y - img_height / 2) / (img_height / 2)
        centrality_score = 1 - (dist_x_norm + dist_y_norm) / 2 # 1 for perfectly central, 0 for edges

        # Combine area and centrality for a simple score. You can adjust weights.
        score = area * centrality_score
        
        scored_locations.append({'location': (top, right, bottom, left), 'score': score})
        
    # Sort by score in descending order
    return sorted(scored_locations, key=lambda x: x['score'], reverse=True)


def process_student_images(source_dir, target_dir, class_name, section):
    # Create target directory if it doesn't exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    student_folders = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    if not student_folders:
        print(f"No student folders found in '{source_dir}'. Please check the path.")
        return

    print(f"Found {len(student_folders)} student folders in '{source_dir}'.")

    for student_folder_name in student_folders:
        student_path = os.path.join(source_dir, student_folder_name)
        
        # Extract name and roll_no from folder name (e.g., Waleed_ur_Rehman_10151)
        try:
            parts = student_folder_name.split('_')
            if len(parts) < 2 or not parts[-1].isdigit():
                print(f"Skipping folder '{student_folder_name}': Invalid naming format. Expected Name_RollNo")
                continue
            
            roll_no = parts[-1]
            name_parts = parts[:-1]
            name_with_spaces = " ".join(name_parts) # Keep spaces for the actual name

        except Exception as e:
            print(f"Error parsing folder name '{student_folder_name}': {e}. Skipping.")
            continue

        print(f"\nProcessing student: {name_with_spaces} (Roll No: {roll_no})")
        
        student_images = [f for f in os.listdir(student_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not student_images:
            print(f"  No images found for {name_with_spaces} in '{student_folder_name}'. Skipping.")
            continue
        
        selected_images_for_student = []
        processed_image_info = [] # Store {image_path, score, original_filename}

        # Process all images in the student folder to find the best ones
        for img_name in student_images:
            img_path = os.path.join(student_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"  Warning: Could not read image '{img_name}'. Skipping.")
                    continue

                # --- Enhancement step ---
                img = enhance_image(img)

                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_img, model="hog")
                scored_locations = filter_and_score_face_locations(face_locations, rgb_img.shape)

                if scored_locations:
                    processed_image_info.append({
                        'path': img_path,
                        'score': scored_locations[0]['score'],
                        'original_filename': img_name
                    })
                else:
                    print(f"  No clear face detected in '{img_name}'.")

            except Exception as e:
                print(f"  Error processing image '{img_name}': {e}")

        # Sort all processed images by score and select the top N
        processed_image_info.sort(key=lambda x: x['score'], reverse=True)
        selected_images_for_student = processed_image_info[:NUM_IMAGES_TO_SELECT]

        if not selected_images_for_student:
            print(f"  No suitable images with clear faces found for {name_with_spaces}. Skipping student.")
            continue

        print(f"  Selected {len(selected_images_for_student)} clear images for {name_with_spaces}.")

        # Copy and rename selected images and create metadata files
        for i, img_info in enumerate(selected_images_for_student):
            original_path = img_info['path']
            
            # Filename for the image (no spaces in name part for path compatibility)
            base_filename_no_spaces = f"{roll_no}_{name_with_spaces.replace(' ', '')}_{class_name.replace(' ', '')}_{section}"
            file_extension = Path(original_path).suffix
            image_new_filename = f"{base_filename_no_spaces}_{i}{file_extension}"
            image_destination_path = os.path.join(target_dir, image_new_filename)
            
            # Metadata for the JSON file (keeps spaces for name)
            metadata = {
                "roll_no": roll_no,
                "name": name_with_spaces,
                "class_name": class_name,
                "section": section,
                "original_filename": img_info['original_filename']
            }
            metadata_new_filename = f"{base_filename_no_spaces}_{i}.json"
            metadata_destination_path = os.path.join(target_dir, metadata_new_filename)
            
            try:
                # Save the enhanced image instead of copying the original
                img = cv2.imread(original_path)
                img = enhance_image(img)
                cv2.imwrite(image_destination_path, img)
                with open(metadata_destination_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                print(f"    Saved enhanced '{img_info['original_filename']}' as '{image_new_filename}' and created metadata.")
            except Exception as e:
                print(f"    Error saving enhanced '{img_info['original_filename']}' or creating metadata: {e}")

    print("\nImage preparation completed!")
    print(f"Please check the '{TARGET_PROCESSED_IMAGES_DIR}' directory and then run 'python upload_students.py'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare student images for bulk upload to FastAPI backend.")
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Path to the directory containing student subfolders (e.g., './Student_Photos').")
    parser.add_argument("--class_name", type=str, required=True,
                        help="The class name for these students (e.g., 'BSCS 8th').")
    parser.add_argument("--section", type=str, required=True,
                        help="The section for these students (e.g., 'B').")
    
    args = parser.parse_args()

    # Assign parsed arguments to configuration variables
    SOURCE_RAW_IMAGES_DIR = args.source_dir
    
    process_student_images(SOURCE_RAW_IMAGES_DIR, TARGET_PROCESSED_IMAGES_DIR, args.class_name, args.section) 