import os
import requests
from pathlib import Path
import json

# Configuration
BACKEND_URL = "http://localhost:8000/admin/students/register"
IMAGES_DIR = "student_images"  # Directory containing processed student images and metadata

def upload_student_images():
    # Create images directory if it doesn't exist
    if not os.path.exists(IMAGES_DIR):
        print(f"Error: {IMAGES_DIR} directory not found! Please run prepare_student_data.py first.")
        return

    # Get all image files from the directory (will ignore JSON files)
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No image files found in {IMAGES_DIR} directory! Please run prepare_student_data.py first.")
        return

    print(f"Found {len(image_files)} image files to upload.")

    # Process each image file
    for image_file in image_files:
        image_path = os.path.join(IMAGES_DIR, image_file)
        
        # Construct the path to the corresponding JSON metadata file
        base_filename_without_ext = Path(image_file).stem
        metadata_file = base_filename_without_ext + ".json"
        metadata_path = os.path.join(IMAGES_DIR, metadata_file)

        if not os.path.exists(metadata_path):
            print(f"Skipping {image_file}: Corresponding metadata file '{metadata_file}' not found.")
            continue

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            roll_no = metadata['roll_no']
            name = metadata['name']
            class_name = metadata['class_name']
            section = metadata['section']

            # Prepare the request
            files = {
                'images': (image_file, open(image_path, 'rb'), 'image/jpeg')
            }
            data = {
                'roll_no': roll_no,
                'name': name,
                'class_name': class_name,
                'section': section
            }

            # Send request to backend
            print(f"\nUploading {name} (Roll No: {roll_no}) with image '{image_file}'...")
            response = requests.post(BACKEND_URL, files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Success: {result['message']}")
            else:
                print(f"Error uploading {name}: {response.text}")

        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in metadata file '{metadata_file}'. Skipping {image_file}.")
        except KeyError as e:
            print(f"Error: Missing key in metadata file '{metadata_file}': {e}. Skipping {image_file}.")
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
        finally:
            # Ensure image file is closed
            if 'files' in locals() and 'images' in files:
                files['images'][1].close()

if __name__ == "__main__":
    print("Starting student image upload process...")
    upload_student_images()
    print("\nUpload process completed!") 