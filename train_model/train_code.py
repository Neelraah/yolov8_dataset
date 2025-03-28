from ultralytics import YOLO
import os
import pandas as pd
from datetime import datetime
import torch
import cv2
import urllib.request
import numpy as np

# Load YOLOv8 model (small version)
model = YOLO('yolov8s.pt')

# Define device (force CPU)
device = torch.device("cpu")
print(f"Using device: {device}")

# Move model to device
model.to(device)

# Function to train the model
def train_model():
    print(f"Training using provided dataset parameters")
    model.train(data='C:/Users/HARLEEN/yolov8_dataset/data.yaml', epochs=10, batch=16, imgsz=640, val=True)
    model.save('yolov8s_traffic.pt')
    print("Training complete. Model saved as yolov8s_traffic.pt")

# Save the model to a separate file
def save_model():
    model.export(format='torchscript', path='saved_models/yolov8s_traffic_scripted.pt')
    print("Model exported and saved to 'saved_models/yolov8s_traffic_scripted.pt')")

# Function to connect to ESP32 CAM and run predictions
def esp32_stream(url):
    print("Starting ESP32 video stream...")
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        print("Error: Cannot open stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run prediction
        results = model.predict(frame)

        # Display results
        annotated_frame = results[0].plot()  # Plot bounding boxes
        cv2.imshow('ESP32 Stream', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Train the model and save it
train_model()
save_model()

# Start ESP32 video streaming with static IP
static_ip = 'http://192.168.1.100/video'  # Replace with your actual static IP
esp32_stream(static_ip)

