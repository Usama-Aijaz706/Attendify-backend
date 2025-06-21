import requests
import json

# Configuration
BASE_URL = "http://13.61.100.248:8000"  # Your actual server URL
TEST_ROLL_NO = "10148"  # Use a roll number that exists in your database

def test_attendance_endpoint():
    """Test the new attendance endpoint"""
    
    # Test the GET endpoint
    url = f"{BASE_URL}/attendance/{TEST_ROLL_NO}"
    
    print(f"Testing GET {url}")
    print("-" * 50)
    
    try:
        response = requests.get(url, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            
            # Validate response structure
            if "status" in data and data["status"] == "success":
                print("✅ Endpoint working correctly!")
                print(f"📊 Found {data.get('total_records', 0)} attendance records for roll number {TEST_ROLL_NO}")
            else:
                print("❌ Unexpected response structure")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON response: {e}")

def test_nonexistent_roll_no():
    """Test with a roll number that doesn't exist"""
    
    url = f"{BASE_URL}/attendance/99999"  # Non-existent roll number
    
    print(f"\nTesting GET {url}")
    print("-" * 50)
    
    try:
        response = requests.get(url, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            
            if data.get("total_records", 0) == 0:
                print("✅ Correctly returned empty results for non-existent roll number")
            else:
                print("❌ Unexpected: found records for non-existent roll number")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    print("Testing Attendance API Endpoint")
    print("=" * 50)
    
    # Test with existing roll number
    test_attendance_endpoint()
    
    # Test with non-existent roll number
    test_nonexistent_roll_no()
    
    print("\n" + "=" * 50)
    print("Test completed!") 