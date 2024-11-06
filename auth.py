import cv2
import os
import numpy as np
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
import base64
import secrets
import time

# Paths
FACE_DATA_PATH = "registered_faces/"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Initialize LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Generate and save a device-based encryption key
def generate_device_key(user_dir):
    key = Fernet.generate_key()
    with open(os.path.join(user_dir, "key.key"), "wb") as key_file:
        key_file.write(key)
    return key

# Load the encryption key for a user
def load_device_key(user_dir):
    with open(os.path.join(user_dir, "key.key"), "rb") as key_file:
        return key_file.read()

# Encrypt and save image data
def encrypt_and_save_image(img_path, img_data, fernet):
    encrypted_data = fernet.encrypt(img_data.tobytes())
    with open(img_path, "wb") as file:
        file.write(encrypted_data)

# Decrypt image for processing
def decrypt_image(img_path, fernet):
    with open(img_path, "rb") as file:
        encrypted_data = file.read()
    decrypted_data = fernet.decrypt(encrypted_data)
    return np.frombuffer(decrypted_data, dtype=np.uint8)

# Collect training data for recognition
def collect_training_data():
    images, labels = [], []
    label_map = {}
    
    for label, name in enumerate(os.listdir(FACE_DATA_PATH)):
        user_dir = os.path.join(FACE_DATA_PATH, name)
        if os.path.isdir(user_dir):
            fernet = Fernet(load_device_key(user_dir))
            label_map[label] = name
            for image_file in os.listdir(user_dir):
                if image_file.endswith(".jpg"):
                    img_path = os.path.join(user_dir, image_file)
                    try:
                        img_data = decrypt_image(img_path, fernet)
                        img = np.reshape(img_data, (100, 100))
                        images.append(img)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error decrypting image: {e}")
    
    return images, np.array(labels), label_map

# Train recognizer with existing data
def train_recognizer():
    images, labels, label_map = collect_training_data()
    if not images:
        print("No registered faces found.")
        return False, {}
    face_recognizer.train(images, labels)
    return True, label_map

# Register user with sequential capture and position guidance
def register_user(name):
    if not consent_prompt():
        print("Registration canceled.")
        return

    cam = cv2.VideoCapture(0)
    user_dir = os.path.join(FACE_DATA_PATH, name)
    os.makedirs(user_dir, exist_ok=True)
    
    # Generate and save a device-based encryption key
    fernet = Fernet(generate_device_key(user_dir))
    
    print("Position your face in front of the camera. Capturing images sequentially.")
    count = 0
    capture_delay = 2  # Delay in seconds between captures
    
    while count < 10:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        if len(faces) == 0:
            print("Face not detected. Please adjust your position.")
            cv2.imshow("Register User", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100))
            
            img_path = os.path.join(user_dir, f"{name}_{count}.jpg")
            encrypt_and_save_image(img_path, face_resized, fernet)
            print(f"Encrypted image {count + 1} saved.")
            count += 1
            
            # Display feedback for the capture
            cv2.putText(frame, f"Image {count} captured", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Register User", frame)
            cv2.waitKey(1)
            
            # Wait before capturing the next image
            time.sleep(capture_delay)
            break  # Only capture one face per frame

    cam.release()
    cv2.destroyAllWindows()
    print(f"User {name} registered with {count} images.")

# Recognize registered users
def recognize_user():
    is_trained, label_map = train_recognizer()
    if not is_trained:
        print("Please register users before recognizing.")
        return
    
    cam = cv2.VideoCapture(0)
    print("Position your face for authentication. Press 'q' to quit.")
    
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100))
            label, confidence = face_recognizer.predict(face_resized)
            name = label_map.get(label, "Unknown")
            
            if confidence < 50:
                cv2.putText(frame, f"Authenticated: {name} ({int(confidence)})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                print(f"Authentication successful for {name}.")
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                print("Authentication failed.")
        
        cv2.imshow("Recognize User", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

# Consent prompt
def consent_prompt():
    consent = input("This system captures and stores facial data for authentication. Do you agree? (yes/no): ")
    return consent.lower() == 'yes'

# Main function with menu
def main():
    os.makedirs(FACE_DATA_PATH, exist_ok=True)
    
    while True:
        print("\nMenu:")
        print("1. Register User")
        print("2. Recognize User")
        print("3. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == '1':
            name = input("Enter name for registration: ")
            register_user(name)
        elif choice == '2':
            recognize_user()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
