import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import FaceRecognitionModel
import os
import random
import torch.nn.functional as F
import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

def load_model(model_path, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FaceRecognitionModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def detect_faces(image):
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces, opencv_image

def prepare_face_for_recognition(opencv_image, face_coords):
    x, y, w, h = face_coords
    face_image = opencv_image[y:y+h, x:x+w]
    
    face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    face_tensor = transform(face_pil).unsqueeze(0)
    return face_tensor

def recognize_face(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        return prediction.item(), confidence.item() * 100

def draw_results(image, faces, predictions, confidences):
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    for (x, y, w, h), pred, conf in zip(faces, predictions, confidences):
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        text = f"ID: {pred} ({conf:.1f}%)"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x, y - 20), (x + text_width, y), (0, 255, 0), -1)
        
        cv2.putText(image, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def process_image(model, device, image_path, true_identity=None):
    image = Image.open(image_path).convert('RGB')
    
    faces, opencv_image = detect_faces(image)
    
    predictions = []
    confidences = []
    
    for face_coords in faces:
        # Prepare face for recognition
        face_tensor = prepare_face_for_recognition(opencv_image, face_coords)
        
        # Recognize face
        prediction, confidence = recognize_face(model, face_tensor, device)
        predictions.append(prediction)
        confidences.append(confidence)
    
    result_image = draw_results(opencv_image, faces, predictions, confidences)
    
    # Display results
    plt.figure(figsize=(12, 8))
    plt.imshow(result_image)
    plt.axis('off')
    
    if true_identity is not None:
        plt.title(f'True Identity: {true_identity}')
    
    plt.show()
    
    return faces, predictions, confidences

def test_random_images(model_path, dataset_path, num_tests=5):
    num_classes = len(os.listdir(os.path.join(dataset_path, 'train')))
    
    model, device = load_model(model_path, num_classes)
    
    test_path = os.path.join(dataset_path, 'test')
    identities = os.listdir(test_path)
    
    print(f"Testing {num_tests} random images...")
    
    for i in range(num_tests):
        identity = random.choice(identities)
        identity_path = os.path.join(test_path, identity)
        images = [f for f in os.listdir(identity_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        image_name = random.choice(images)
        image_path = os.path.join(identity_path, image_name)
        
        print(f"\nTest {i+1}:")
        print(f"Image: {image_name}")
        print(f"True Identity: {identity}")
        
        faces, predictions, confidences = process_image(model, device, image_path, identity)
        
        # Print results for each detected face
        for j, (pred, conf) in enumerate(zip(predictions, confidences)):
            print(f"Face {j+1}:")
            print(f"  Predicted Identity: {pred}")
            print(f"  Confidence: {conf:.2f}%")
            print(f"  Correct: {pred == int(identity)}")

def process_webcam(model_path, dataset_path):
    num_classes = len(os.listdir(os.path.join(dataset_path, 'train')))
    model, device = load_model(model_path, num_classes)
    
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        faces, opencv_image = detect_faces(pil_image)
        
        predictions = []
        confidences = []
        
        for face_coords in faces:
            face_tensor = prepare_face_for_recognition(opencv_image, face_coords)
            prediction, confidence = recognize_face(model, face_tensor, device)
            predictions.append(prediction)
            confidences.append(confidence)
        
        result_frame = draw_results(opencv_image, faces, predictions, confidences)
        
        cv2.imshow('Face Recognition', cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

class FaceRecognitionApp:
    def __init__(self, model_path, num_classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = FaceRecognitionModel(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def prepare_face(self, frame, face_coords):
        x, y, w, h = face_coords
        face_img = frame[y:y+h, x:x+w]
        
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_img)
        
        face_tensor = self.transform(face_pil).unsqueeze(0)
        return face_tensor
    
    def recognize_face(self, face_tensor):
        with torch.no_grad():
            face_tensor = face_tensor.to(self.device)
            output = self.model(face_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            return prediction.item(), confidence.item() * 100
    
    def draw_results(self, frame, faces, predictions, confidences):
        for (x, y, w, h), pred, conf in zip(faces, predictions, confidences):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            text = f"ID: {pred} ({conf:.1f}%)"
            
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x, y - 20), (x + text_width, y), (0, 255, 0), -1)
            
            cv2.putText(frame, text, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        cv2.putText(frame, "Press 'q' to quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        print("Starting webcam face recognition...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read from webcam")
                break
            
            faces = self.detect_faces(frame)
            
            predictions = []
            confidences = []
            
            for face_coords in faces:
                face_tensor = self.prepare_face(frame, face_coords)
                prediction, confidence = self.recognize_face(face_tensor)
                predictions.append(prediction)
                confidences.append(confidence)
            
            frame = self.draw_results(frame, faces, predictions, confidences)
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

    def predict(self, frame):
        faces = self.detect_faces(frame)
        
        results = []
        
        for face_coords in faces:
            face_tensor = self.prepare_face(frame, face_coords)
            prediction, confidence = self.recognize_face(face_tensor)

            results.append({"prediction": prediction, "confidence": confidence})

        return results


from typing import TypedDict
from time import sleep

class Suspect(TypedDict):
    name: str
    threatLevel: str
    crimes: list[str]
    confidence: float

def fetch_suspect_database_by_id(id: int, confidence: float) -> Suspect:
    if id == 0:
        return {"name": "João Matheus", "threatLevel": "CRITICAL", "crimes": ["insider trading", "fraud"], "confidence": confidence}
    elif id == 2:
        return {"name": "Anderson Laurentino", "threatLevel": "CRITICAL", "crimes": ["hacking", "identity theft"], "confidence": confidence}
    elif id == 1:
        return {"name": "Helton Alves", "threatLevel": "CRITICAL", "crimes": ["vandalism", "robbery"], "confidence": confidence}
    else:
        return {"name": "UNKNOWN", "threatLevel": "UNKNOWN", "crimes": [], "confidence": 0}


model_path = './model.pth'
num_classes = 3

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please ensure the model file exists and try again.")

model = FaceRecognitionApp(model_path, num_classes)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-image/")
async def upload_image(image: UploadFile = File(...)):
    contents = await image.read()  # read image bytes
    np_arr = np.frombuffer(contents, np.uint8)  # convert bytes to numpy array
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model.predict(frame)

    sleep(1)

    return JSONResponse([
        fetch_suspect_database_by_id(suspect["prediction"], suspect["confidence"])
        for suspect in results
    ])
