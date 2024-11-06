# Imports
import cv2
import numpy as np
import os
from cryptography.fernet import Fernet
import pickle
import time

# Constants and Paths
DATA_PATH = "./face_data"
KEY_PATH = "./key.key"

# Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Ensure data directory exists
os.makedirs(DATA_PATH, exist_ok=True)

# Generate or load encryption key
def load_key():
    if os.path.exists(KEY_PATH):
        with open(KEY_PATH, "rb") as key_file:
            key = key_file.read()
    else:
        key = Fernet.generate_key()
        with open(KEY_PATH, "wb") as key_file:
            key_file.write(key)
    return key

encryption_key = load_key()
cipher_suite = Fernet(encryption_key)

# Simple feature extractor using histogram of pixel intensities
def extract_features(face_image):
    hist = cv2.calcHist([face_image], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Encrypt and save embedding
def save_encrypted_embedding(user_id, features):
    encrypted_data = cipher_suite.encrypt(pickle.dumps(features))
    with open(os.path.join(DATA_PATH, f"{user_id}.bin"), "wb") as f:
        f.write(encrypted_data)
    print(f"User {user_id} enrolled successfully.")

# Load and decrypt embedding
def load_encrypted_embedding(user_id):
    try:
        with open(os.path.join(DATA_PATH, f"{user_id}.bin"), "rb") as f:
            encrypted_data = f.read()
        features = pickle.loads(cipher_suite.decrypt(encrypted_data))
        return features
    except FileNotFoundError:
        return None

# Compare histograms for authentication
def authenticate(user_id, face_image):
    features = extract_features(face_image)
    stored_features = load_encrypted_embedding(user_id)
    if stored_features is None:
        print(f"No data found for user {user_id}.")
        return False
    
    # Simple similarity check using correlation
    similarity = cv2.compareHist(features, stored_features, cv2.HISTCMP_CORREL)
    if similarity > 0.8:  # Threshold can be adjusted based on testing
        print(f"User {user_id} authenticated successfully.")
        return True
    else:
        print(f"User {user_id} failed authentication.")
        return False


def enroll_user(user_id, num_samples=5, capture_method="manual"):
    cap = cv2.VideoCapture(0)
    print("Please position your face in front of the camera...")

    # Define instructions for different positions
    instructions = [
        "Look straight ahead",
        "Turn your head slightly to the left",
        "Turn your head slightly to the right",
        "Look slightly up",
        "Look slightly down"
    ]

    features_list = []
    current_instruction_index = 0

    while len(features_list) < num_samples:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        # Display instruction on screen
        instruction_text = instructions[current_instruction_index]
        cv2.putText(frame, instruction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for (x, y, w, h) in faces:
            face_image = gray[y:y + h, x:x + w]

            # Option 1: Timer-based capture (wait 2 seconds between each capture)
            if capture_method == "timer":
                print(f"Capturing '{instruction_text}' in 2 seconds...")
                cv2.imshow("Enrollment - Adjust position", frame)
                cv2.waitKey(1)  # Refresh the window to display text
                time.sleep(2)   # Wait for 2 seconds before capture
                features = extract_features(face_image)
                features_list.append(features)
                print(f"Captured sample {len(features_list)} for: '{instruction_text}'")

            # Option 2: User-triggered capture (wait for space bar)
            elif capture_method == "manual":
                print(f"Press the space bar to capture '{instruction_text}'")
                while True:
                    cv2.imshow("Enrollment - Press 'space' to capture", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(" "):  # Space bar pressed
                        features = extract_features(face_image)
                        features_list.append(features)
                        print(f"Captured sample {len(features_list)} for: '{instruction_text}'")
                        break

            # Move to the next instruction after each capture
            current_instruction_index = (current_instruction_index + 1) % len(instructions)

            # Draw rectangle around the face for visual feedback
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Break if we have reached the required number of samples
            if len(features_list) >= num_samples:
                break

        cv2.imshow("Enrollment", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Compute the average feature vector from all positions
    if features_list:
        avg_features = np.mean(features_list, axis=0)
        save_encrypted_embedding(user_id, avg_features)
        print(f"User {user_id} enrolled successfully with samples in various head positions.")

    cap.release()
    cv2.destroyAllWindows()

# Capture face from webcam and authenticate user
def authenticate_user(user_id):
    cap = cv2.VideoCapture(0)
    print("Please position your face in front of the camera...")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_image = gray[y:y + h, x:x + w]
            if authenticate(user_id, face_image):
                cap.release()
                return True
            else:
                cap.release()
                return False

        cv2.imshow("Authentication - Press 'q' to exit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False

# Menu
def main():
    while True:
        print("\nFace Recognition System")
        print("1. Enroll User")
        print("2. Authenticate User")
        print("3. Exit")

        choice = input("Enter your choice: ")
        if choice == "1":
            user_id = input("Enter user ID to enroll: ")
            enroll_user(user_id)
        elif choice == "2":
            user_id = input("Enter user ID to authenticate: ")
            authenticated = authenticate_user(user_id)
            if authenticated:
                print("Access Granted.")
            else:
                print("Access Denied.")
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
